"""Shared Constants - UI와 Backend에서 공유되는 상수"""

# 이미지 분류 카테고리
IMAGE_CATEGORIES = {
    "people": "인물",
    "nature": "꽃 또는 경치",
    "text": "글자 등 문자포함",
    "events": "행사",
}

# 카테고리 설명
CATEGORY_DESCRIPTIONS = {
    "people": "사람이 포함된 이미지",
    "nature": "꽃, 나무, 풍경 등 자연 이미지",
    "text": "문서, 책, 글씨 등 텍스트가 포함된 이미지",
    "events": "여행, 생일, 산책 등 행사 이미지",
}

# 지원 이미지 확장자
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}

# 신뢰도 임계값
CONFIDENCE_THRESHOLD = 0.5
