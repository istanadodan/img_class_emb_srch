"""Image Classifier Service - 이미지 분류 비즈니스 로직"""

import asyncio
from pathlib import Path
from typing import List, Optional
from backend.clients.base import AIClient
from backend.domain.models import Image, ClassificationResult
from backend.system.exceptions import ImageProcessingException


class ImageClassifierService:
    """이미지 분류 서비스"""

    def __init__(self, ai_client: AIClient):
        """
        Args:
            ai_client: 사용할 AI 클라이언트
        """
        self.ai_client = ai_client

    async def classify_single_image(self, image_path: str) -> ClassificationResult:
        """
        단일 이미지 분류

        Args:
            image_path: 이미지 파일 경로

        Returns:
            분류 결과
        """
        try:
            result = await self.ai_client.classify_image(image_path)
            return ClassificationResult(
                image_path=image_path,
                category=result["category"],
                confidence=result["confidence"],
                description=result.get("description", ""),
            )
        except Exception as e:
            raise ImageProcessingException(f"Failed to classify image: {str(e)}")

    async def classify_batch(
        self, image_paths: List[str], skip_errors: bool = True
    ) -> tuple[List[ClassificationResult], List[tuple[str, str]]]:
        """
        배치 이미지 분류

        Args:
            image_paths: 이미지 파일 경로 리스트
            skip_errors: True이면 오류 스킵, False이면 첫 오류에서 중단

        Returns:
            (분류 결과 리스트, 오류 리스트)
        """
        results = []
        errors = []

        for image_path in image_paths:
            try:
                result = await self.classify_single_image(image_path)
                results.append(result)
            except Exception as e:
                error_msg = str(e)
                errors.append((image_path, error_msg))
                if not skip_errors:
                    raise

        return results, errors

    async def classify_folder(
        self, folder_path: str, skip_errors: bool = True
    ) -> tuple[List[ClassificationResult], List[tuple[str, str]]]:
        """
        폴더 내 모든 이미지 분류

        Args:
            folder_path: 폴더 경로
            skip_errors: True이면 오류 스킵, False이면 첫 오류에서 중단

        Returns:
            (분류 결과 리스트, 오류 리스트)
        """
        folder = Path(folder_path)
        if not folder.is_dir():
            raise ImageProcessingException(f"Not a directory: {folder_path}")

        # 지원하는 이미지 확장자
        image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
        image_paths = [
            str(p)
            for p in folder.rglob("*")
            if p.is_file() and p.suffix.lower() in image_extensions
        ]

        if not image_paths:
            raise ImageProcessingException(f"No images found in: {folder_path}")

        return await self.classify_batch(image_paths, skip_errors=skip_errors)
