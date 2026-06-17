"""Image Classifier Service - 이미지 분류 비즈니스 로직"""

import io
import asyncio
from pathlib import Path
from typing import List, Optional, Any
from PIL import Image
from backend.clients.base import AIClient
from shared.classification_schema import ClassificationResponse
from backend.system.exceptions import ImageProcessingException
from backend.service.graph.workflow import create_classification_workflow
from backend.system.event_handler import UIEventHandler

from backend.clients.llm_clients import StudioLLMClient # Updated import

class ImageClassifierService:
    """이미지 분류 서비스"""

    def __init__(self, llm_client: AIClient, embedding_client: Optional[Any] = None):
        """
        Args:
            ai_client: 분류용 AI 클라이언트
            embedding_client: 임베딩 생성 클라이언트 (선택사항)
        """
        # 워크플로우 인스턴스를 서비스 생성 시 1회만 생성하여 재사용 (성능 최적화)
        self.workflow = create_classification_workflow(llm_client, embedding_client)

    async def classify_single_image(
        self, image_path: str, event_queue: asyncio.Queue
    ) -> ClassificationResponse:
        """
        단일 이미지 분류
        """
        results, errors = await self.classify_batch([image_path], event_queue)
        if errors:
            # 배치 처리 중 에러 발생 시 (로딩 실패 등) 에러 정보 포함 결과 반환
            return ClassificationResponse(
                image_path=image_path,
                errors=f"Classification failed: {errors[0][1]}",
            )
        return results[0]

    async def classify_batch(
        self,
        image_paths: List[str],
        event_queue: asyncio.Queue,
        skip_errors: bool = True,
    ) -> tuple[List[ClassificationResponse], List[tuple[str, str]]]:
        """
        배치 이미지 분류 (단일 워크플로우 인스턴스 내 병렬 처리)
        """
        results = []
        errors = []
        # 이벤트 큐에 총 개수 전달
        await event_queue.put({"type": "total_count", "total_count": len(image_paths)})

        # 1. 이미지 전처리 병렬 수행 (I/O & CPU)
        await UIEventHandler.publish(
            event_queue, "status", f"{len(image_paths)}개 이미지 로드 중..."
        )

        load_tasks = [asyncio.to_thread(self._load_and_resize_image, path) for path in image_paths]
        try:
            loaded_images = await asyncio.wait_for(
                asyncio.gather(
                    *load_tasks,
                    return_exceptions=True,
                ),
                timeout=15,
            )
        except asyncio.TimeoutError as e:
            await UIEventHandler.publish(event_queue, "error", f"로드 시간 초과: {str(e)}")
            return results, [(path, f"로드 시간 초과: {str(e)}") for path in image_paths]

        # 2. 유효한 데이터로 초기 상태 리스트 준비
        initial_states = []

        for i, img in enumerate(loaded_images):
            if isinstance(img, Exception):
                err_msg = f"Load error: {str(img)}"
                errors.append((image_paths[i], err_msg))
                await UIEventHandler.publish(event_queue, "error", err_msg, path=image_paths[i])
                if not skip_errors:
                    raise img
                continue

            initial_states.append(
                {
                    "image_path": image_paths[i],
                    "image_bytes": img,
                    "category": None,
                    "confidence": 0.0,
                    "description": "",
                    "objects": [],
                    "embedding": None,
                    "error": None,
                    "retry_prompt": "",
                    "max_retires": 3,
                    "retry_cnt": 0,
                    "logs": [],
                }
            )

        if not initial_states:
            return results, errors

        # 3. 단일 워크플로우 인스턴스에서 abatch를 통한 병렬 처리
        await UIEventHandler.publish(
            event_queue, "status", f"{len(initial_states)}개 이미지 분석 시작 (병렬)..."
        )

        from langchain_core.runnables import RunnableConfig

        # RunnableConfig를 통해 event_queue 전달 (노드에서 사용 가능)
        config = RunnableConfig(max_concurrency=10, configurable={"event_queue": event_queue})

        batch_results = await self.workflow.abatch(initial_states, config=config)

        # 4. 결과 매핑 및 최종 에러 확인
        for state in batch_results:
            img_path = state["image_path"]
            error = state.get("error")
            category = state.get("category") or "no-class"

            if error:
                errors.append((img_path, error))

            res = ClassificationResponse(
                image_path=img_path,
                category=category,
                confidence=state.get("confidence", 0.0),
                description=state.get("description", "Auto-classified"),
                objects=state.get("objects", []),
                errors=error or "",
            )
            results.append(res)

        await UIEventHandler.publish(event_queue, "status", "모든 이미지 처리 완료")
        return results, errors

    async def classify_folder(
        self,
        folder_path: str,
        event_queue: asyncio.Queue,
        skip_errors: bool = True,
    ) -> tuple[List[ClassificationResponse], List[tuple[str, str]]]:
        """
        폴더 내 모든 이미지 분류
        """
        folder = Path(folder_path)
        if not folder.is_dir():
            raise ImageProcessingException(f"Not a directory: {folder_path}")

        image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
        image_paths = [
            str(p) for p in folder.glob("*") if p.is_file() and p.suffix.lower() in image_extensions
        ]

        if not image_paths:
            raise ImageProcessingException(f"No images found in: {folder_path}")

        return await self.classify_batch(
            image_paths, skip_errors=skip_errors, event_queue=event_queue
        )

    def _load_and_resize_image(self, image_path: str) -> bytes:
        """이미지 로드 및 리사이징 동기 함수 (스레드용)"""
        img_bytes = io.BytesIO()

        img = Image.open(image_path)

        # 이미지 리사이징 (최대 1024x1024 유지)
        img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)

        # RGBA 등의 모드를 RGB로 통일
        if img.mode != "RGB":
            img = img.convert("RGB")

        # 실제 데이터를 메모리에 로드하기 위해 로드 강제 (지연 로딩 방지)
        img.save(img_bytes, format="JPEG", quality=65, optimize=True)
        # img.load() -> return img (PIL.Image.Image)
        return img_bytes.getvalue()
