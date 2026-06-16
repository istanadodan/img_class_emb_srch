"""Database Models - SQLAlchemy ORM 모델"""

from datetime import datetime, UTC
from typing import Optional
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, JSON, func
from sqlalchemy.orm import mapped_column, Mapped, relationship
from pgvector.sqlalchemy import Vector
from backend.system.database import Base


class ImageRecord(Base):
    """이미지 기록 (데이터베이스 모델)"""

    __tablename__ = "images"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    path: Mapped[str] = mapped_column(unique=True, index=True)
    category: Mapped[str] = mapped_column(index=True)
    confidence: Mapped[float] = mapped_column()
    description: Mapped[str] = mapped_column()
    objects: Mapped[str] = mapped_column(default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )

    embeddings = relationship(
        "EmbeddingRecord", back_populates="image", cascade="all, delete-orphan"
    )


class EmbeddingRecord(Base):
    """임베딩 벡터 기록 (데이터베이스 모델)"""

    __tablename__ = "embeddings"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    image_id: Mapped[int] = mapped_column(ForeignKey("images.id", ondelete="CASCADE"), index=True)
    model_name: Mapped[str] = mapped_column(default="CLIP-ViT-B-32")
    vector_dim: Mapped[int] = mapped_column(default=4096)
    vector: Mapped[list[float]] = mapped_column(Vector(4096))  # pgvector 4096차원 벡터
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    image = relationship("ImageRecord", back_populates="embeddings")
