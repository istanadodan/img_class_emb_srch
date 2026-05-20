"""Domain Models - 이미지 분류 관련 도메인 객체"""

from enum import Enum
from typing import Optional
from pydantic import BaseModel


class ImageCategory(str, Enum):
    """이미지 분류 카테고리"""

    PEOPLE = "people"  # 인물
    NATURE = "nature"  # 꽃 또는 경치
    TEXT = "text"  # 글자 등 문자포함
    EVENTS = "events"  # 행사 (여행, 생일, 산책 등)


class Image(BaseModel):
    """이미지 도메인 객체"""

    path: str
    category: Optional[ImageCategory] = None
    confidence: float = 0.0
    metadata: dict = {}

    class Config:
        use_enum_values = True


class ClassificationResult(BaseModel):
    """분류 결과 도메인 객체"""

    image_path: str
    category: ImageCategory
    confidence: float
    description: str = ""

    class Config:
        use_enum_values = True
