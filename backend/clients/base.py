"""Base AI Client - 모든 AI 클라이언트의 기본 인터페이스"""

from abc import ABC, abstractmethod
from shared.classification_schema import ClassificationResult
from typing import Optional, Any


class AIClient(ABC):
    """AI 모델 클라이언트 기본 인터페이스"""

    @abstractmethod
    def get_model(self) -> str:
        """사용 중인 모델 이름 반환"""

    @abstractmethod
    async def classify_image(self, image_data: bytes) -> ClassificationResult:
        """
        이미지를 분류하고 결과 반환

        Args:
            image_path: 이미지 파일 경로
            image_data: 이미 로드된 이미지 객체(PIL.Image 등) (선택 사항)

        Returns:
            분류 결과 (ClassificationResult 모델)
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """클라이언트 이름"""
        pass
