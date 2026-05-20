"""Base AI Client - 모든 AI 클라이언트의 기본 인터페이스"""

from abc import ABC, abstractmethod
from backend.domain.models import ImageCategory


class AIClient(ABC):
    """AI 모델 클라이언트 기본 인터페이스"""

    @abstractmethod
    async def classify_image(self, image_path: str) -> dict:
        """
        이미지를 분류하고 결과 반환

        Args:
            image_path: 이미지 파일 경로

        Returns:
            {
                "category": ImageCategory,
                "confidence": float,
                "description": str (선택사항)
            }
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """클라이언트 사용 가능 여부 확인"""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """클라이언트 이름"""
        pass
