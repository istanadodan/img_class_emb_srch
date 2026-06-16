from backend.system.database import transactional
from typing import Optional, Any
from sqlalchemy.orm import Session  # Import Session
from backend.service.vector_search import VectorSearchService
from pathlib import Path


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
    def get_all_images(
        self, categories: Optional[list[str]] = None, db: Optional[Session] = None
    ) -> list[dict]:
        return self.vector_service.get_all_images(categories=categories, db=db)
