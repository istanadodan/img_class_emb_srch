"""Vector Search Service - 벡터 기반 유사 이미지 검색 (pgvector 최적화)"""

import logging
from typing import Optional
from sqlalchemy.orm import Session
from backend.clients.embedding_client import EmbeddingClient
from backend.domain.db_models import ImageRecord, EmbeddingRecord
from backend.system.exceptions import AIClientException
from backend.system.database import transactional

logger = logging.getLogger(__name__)


class VectorSearchService:
    """벡터 검색 서비스"""

    def __init__(self, embedding_client: EmbeddingClient):
        """
        Args:
            embedding_client: 임베딩 생성 클라이언트
        """
        self.embedding_client = embedding_client

    @transactional
    async def save_embedding(
        self,
        image_path: str,
        embedding: list[float],
        category: str,
        confidence: float,
        description: str,
        objects: list[str],
        db: Optional[Session] = None,
    ) -> EmbeddingRecord:
        """
        이미지의 임베딩 벡터를 생성하고 데이터베이스에 저장

        Args:
            db: 데이터베이스 세션
            image_path: 이미지 파일 경로
            category: 분류 카테고리
            confidence: 신뢰도
            description: 설명
            objects: 감지된 객체 리스트
            model_name: 임베딩 모델 이름

        Returns:
            저장된 임베딩 레코드
        """
        if not db:
            raise AIClientException("Database session is required for search")

        # 기존 이미지 레코드 확인 및 생성
        image_record: ImageRecord = db.query(ImageRecord).filter_by(path=image_path).first()
        if not image_record:
            image_record = ImageRecord(
                path=image_path,
                category=category,
                confidence=confidence,
                description=description,
                objects=",".join(objects),
            )
            db.add(image_record)
            db.flush()  # Flush to get image_record.id
        else:
            # 기존 레코드 업데이트
            image_record.category = category
            image_record.confidence = confidence
            image_record.description = description
            image_record.objects = ",".join(objects)
            db.query(EmbeddingRecord).filter_by(image_id=image_record.id).delete()
            db.flush()  # Flush to ensure delete is processed before new embedding

        # pgvector 저장 (List[float] 형식)
        embedding_record = EmbeddingRecord(
            image_id=image_record.id,
            model_name=self.embedding_client.get_model(),
            vector=embedding,  # pgvector에 직접 저장
            vector_dim=len(embedding),
        )
        db.add(embedding_record)

        return embedding_record

    @transactional
    def search_similar_images(
        self,
        image_path: str,
        top_k: int = 5,
        category_filter: Optional[str] = None,
        db: Optional[Session] = None,
    ) -> list[dict]:
        """
        유사 이미지 검색 (PostgreSQL pgvector 코사인 거리)

        Args:
            db: 데이터베이스 세션
            image_path: 쿼리 이미지 경로
            top_k: 반환할 상위 K개 이미지
            category_filter: 카테고리 필터 (선택사항)

        Returns:
            유사 이미지 목록 [{"path": str, "similarity": float, ...}]
        """
        if not db:
            raise AIClientException("Database session is required for search")

        # 쿼리 이미지의 임베딩 가져오기
        query_image = db.query(ImageRecord).filter_by(path=image_path).first()
        if not query_image:
            raise AIClientException(f"Image not found: {image_path}")

        query_embedding = db.query(EmbeddingRecord).filter_by(image_id=query_image.id).first()
        if not query_embedding:
            raise AIClientException(f"Embedding not found for: {image_path}")

        # pgvector 코사인 거리 계산 (cosine_distance 메서드 사용)
        similarity_score = 1 - EmbeddingRecord.vector.cosine_distance(query_embedding.vector)

        # 쿼리 작성
        query = (
            db.query(
                ImageRecord.id,
                ImageRecord.path,
                ImageRecord.category,
                ImageRecord.confidence,
                ImageRecord.description,
                ImageRecord.objects,
                similarity_score.label("similarity"),
            ).join(EmbeddingRecord, ImageRecord.id == EmbeddingRecord.image_id)
            # .filter(ImageRecord.id != query_image.id)
        )  # 자신 제외

        # 카테고리 필터 적용
        if category_filter:
            query = query.filter(ImageRecord.category == category_filter)

        # 유사도 임계값 적용. 60% 초과
        query = query.filter(similarity_score > 0.6)
        # 유사도 순으로 정렬 및 상위 K개 조회
        results = query.order_by(similarity_score.desc()).limit(top_k).all()

        # 결과를 딕셔너리로 변환
        similarities = []
        for row in results:
            similarities.append(
                {
                    "id": row.id,
                    "path": row.path,
                    "category": row.category,
                    "confidence": row.confidence,
                    "description": row.description,
                    "objects": row.objects,
                    "similarity": float(row.similarity),
                }
            )

        return similarities

    @transactional
    def get_all_images(
        self,
        categories: Optional[list[str]] = None,
        db: Optional[Session] = None,
    ) -> list[dict]:
        """
        데이터베이스의 모든 이미지 목록 조회

        Args:
            db: 데이터베이스 세션
            categories: 필터링할 카테고리 리스트 (선택사항)

        Returns:
            이미지 목록
        """
        if not db:
            raise AIClientException("Database session is required")

        query = db.query(ImageRecord)

        if categories:
            query = query.filter(ImageRecord.category.in_(categories))

        results = query.order_by(ImageRecord.id.desc()).all()

        return [
            {
                "id": row.id,
                "path": row.path,
                "category": row.category,
                "confidence": row.confidence,
                "description": row.description,
                "objects": row.objects,
            }
            for row in results
        ]

    @transactional
    def get_category_similar_images(
        self,
        image_path: str,
        top_k: int = 5,
        db: Optional[Session] = None,
    ) -> dict[str, list[dict]]:
        """
        카테고리별로 유사 이미지 검색

        Args:
            db: 데이터베이스 세션
            image_path: 쿼리 이미지 경로
            top_k: 카테고리당 반환할 이미지 수

        Returns:
            카테고리별 유사 이미지 딕셔너리
        """
        results = {}
        for category in ["people", "nature", "text", "events"]:
            # Note: self.search_similar_images is also decorated,
            # but we pass the 'db' session explicitly here to ensure it's reused
            similar = self.search_similar_images(
                image_path, top_k=top_k, category_filter=category, db=db
            )
            if similar:
                results[category] = similar

        return results
