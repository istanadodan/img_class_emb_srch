"""Presentation Schemas - API 응답 및 요청 스키마"""

from typing import List
from pydantic import BaseModel
from backend.domain.models import ImageCategory


class ClassificationResponse(BaseModel):
    """이미지 분류 응답 스키마"""

    image_path: str
    category: ImageCategory
    confidence: float
    description: str = ""

    class Config:
        use_enum_values = True


class BatchClassificationResponse(BaseModel):
    """배치 분류 응답 스키마"""

    total_images: int
    successful: int
    failed: int
    results: List[ClassificationResponse]


class ErrorResponse(BaseModel):
    """에러 응답 스키마"""

    error: str
    detail: str = ""
    code: int = 400
