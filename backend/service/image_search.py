from backend.system.database import transactional
from typing import Optional, Any, List, cast
from sqlalchemy.orm import Session
from backend.service.vector_search import VectorSearchService
from backend.domain.db_models import ImageRecord
from pathlib import Path
import logging

from backend.clients.agent_executor_client import get_agent_executor_client, AgentExecutorClient

# No need for BaseMessage, ToolCall from langchain_core as AgentExecutorClient handles execution directly.

logger = logging.getLogger(__name__)


class ImgSearchService:

    def __init__(self, embedding_client):
        self.vector_service = VectorSearchService(embedding_client=embedding_client)

    @transactional
    def search_by_path(
        self, image_path: str, category_filter: Optional[str] = None, db: Optional[Session] = None
    ) -> list[dict]:
        # 경로 전처리: 따옴표 제거 및 절대 경로 변환
        clean_path = str(Path(image_path.replace('"', "")).resolve())

        # 파일 존재 여부 확인
        if not Path(clean_path).is_file():
            from backend.system.exceptions import ImageProcessingException

            raise ImageProcessingException(f"파일을 찾을 수 없습니다: {clean_path}")

        results = self.vector_service.search_similar_images(
            image_path=clean_path, top_k=5, category_filter=category_filter, db=db
        )
        return results

    @transactional
    async def search_by_text(
        self, query_text: str, category_filter: Optional[str] = None, db: Optional[Session] = None
    ) -> list[dict]:
        """
        자연어 쿼리를 LangChain AgentExecutor를 통해 처리하여 유사 이미지를 검색
        """
        agent_executor_client: AgentExecutorClient = get_agent_executor_client()

        # 1. AgentExecutor에게 쿼리를 전달하고 구조화된 결과(List[dict])를 직접 받음
        structured_results: List[dict] = await agent_executor_client.ask(query_text)

        # 2. 결과를 UI에서 기대하는 형식으로 매핑
        if not structured_results:
            return []

        final_results = []
        for res in structured_results:
            image_id = res.get("image_id")
            # 유사도 정규화 (LLM이 0~100 사이의 값을 줄 경우 대비)
            raw_similarity = res.get("similarity", 0.0)
            if raw_similarity > 1.0:
                raw_similarity /= 100.0
            similarity = min(max(float(raw_similarity), 0.0), 1.0)

            # DB에서 원본 정보 조회
            image_record = (
                db.query(ImageRecord).filter_by(id=image_id).first() if db else None
            )

            final_results.append(
                {
                    "id": image_id,
                    "path": res.get("path"),
                    "similarity": similarity,
                    "justification": res.get("justification"),  # AI 에이전트의 개별 판단 근거
                    "category": image_record.category if image_record else "AI Search",
                    "description": image_record.description
                    if image_record
                    else "AI 에이전트 검색 결과",
                    "confidence": image_record.confidence if image_record else 1.0,
                    "objects": image_record.objects if image_record else "",
                }
            )

        return final_results

    @transactional
    def get_all_images(
        self, categories: Optional[list[str]] = None, db: Optional[Session] = None
    ) -> list[dict]:
        return self.vector_service.get_all_images(categories=categories, db=db)
