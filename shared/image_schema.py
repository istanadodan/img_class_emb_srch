"""Domain Models - 이미지 분류 관련 도메인 객체"""

from enum import Enum
from typing import Optional, Literal, Any
from pydantic import BaseModel, ConfigDict


class ImageCategory(str, Enum):
    """이미지 분류 카테고리"""

    PEOPLE = "people"  # 인물
    PLACE = "place"
    ANIMAL = "animal"
    FLOWER = "flower"  # 꽃
    NATURE = "nature"  # 경치 등 자연
    TEXT = "text"  # 글자 등 문자포함
    EVENTS = "events"  # 행사 (여행, 생일, 산책 등)
    NoCLASS = "no-class"


class Image(BaseModel):
    """이미지 도메인 객체"""

    path: str
    category: Optional[ImageCategory] = None
    confidence: float = 0.0
    metadata: dict = {}

    model_config = ConfigDict(use_enum_values=True)