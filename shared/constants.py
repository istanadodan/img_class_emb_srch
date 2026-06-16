"""Shared Constants - UI와 Backend에서 공유되는 상수"""

# 이미지 분류 카테고리
IMAGE_CATEGORIES = {
    "people": "인물",
    "animal": "동물",
    "place": "장소",
    "flower": "꽃",
    "nature": "경치",
    "text": "텍스트",
    "events": "행사",
}

# 카테고리 설명
CATEGORY_DESCRIPTIONS = {
    "people": "사람이 중심이 되거나 포함된 이미지",
    "animal": "강아지, 고양이, 새, 곤충 등 동물 이미지",
    "place": "건물, 거리, 도시 등 장소나 환경 이미지",
    "flower": "꽃, 화초 등 식물의 꽃이 주가 되는 이미지",
    "nature": "나무, 산, 바다, 하늘 등 자연 경치 이미지",
    "text": "문서, 책, 글씨 등 텍스트가 명확히 포함된 이미지",
    "events": "여행, 생일, 산책, 축제 등 특정 행사나 활동 이미지",
}

# 지원 이미지 확장자
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}

# 신뢰도 임계값
CONFIDENCE_THRESHOLD = 0.5
