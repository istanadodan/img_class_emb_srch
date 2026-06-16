"""Presentation Schemas - API 응답 및 요청 스키마"""

from typing import List
from pydantic import BaseModel
from shared.image_schema import ImageCategory
from typing import Optional, Literal, Any
from pydantic import BaseModel, ConfigDict


class ClassificationResult(BaseModel):
    """분류 결과 도메인 객체"""

    category: ImageCategory | Literal["no-class"]
    confidence: float
    description: str
    objects: list[str] = []

    model_config = ConfigDict(use_enum_values=True, arbitrary_types_allowed=True)


class ClassificationResponse(BaseModel):
    """이미지 분류 응답 스키마"""

    image_path: str
    description: str = ""
    category: ImageCategory | Literal["no-class"] = "no-class"
    confidence: float = 0.0
    objects: list[str] = []
    errors: str = ""

    model_config = ConfigDict(use_enum_values=True, arbitrary_types_allowed=True)


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
