"""StudioLM AI Client - Local StudioLM API를 통한 이미지 분류"""

import httpx
from typing import Optional
from backend.clients.base import AIClient
from backend.system.config import settings
from backend.system.exceptions import AIClientException
from backend.domain.models import ImageCategory


class StudioLMClient(AIClient):
    """StudioLM API 클라이언트"""

    def __init__(
        self,
        api_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: int = 30,
    ):
        self.api_url = api_url or settings.studiolm_api_url
        self.api_key = api_key or settings.studiolm_api_key
        self.timeout = timeout
        self._client = None

    @property
    def name(self) -> str:
        return "StudioLM"

    def is_available(self) -> bool:
        """StudioLM API 서버 연결 가능 여부 확인"""
        try:
            response = httpx.get(f"{self.api_url}/health", timeout=5)
            return response.status_code == 200
        except Exception:
            return False

    async def classify_image(self, image_path: str) -> dict:
        """
        이미지를 StudioLM API로 분류

        Args:
            image_path: 이미지 파일 경로

        Returns:
            분류 결과 딕셔너리
        """
        try:
            with open(image_path, "rb") as image_file:
                files = {"image": image_file}
                headers = {}
                if self.api_key:
                    headers["Authorization"] = f"Bearer {self.api_key}"

                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(
                        f"{self.api_url}/classify",
                        files=files,
                        headers=headers,
                    )

                if response.status_code != 200:
                    raise AIClientException(
                        f"StudioLM API error: {response.status_code} - {response.text}"
                    )

                result = response.json()
                return {
                    "category": ImageCategory(result.get("category")),
                    "confidence": float(result.get("confidence", 0.0)),
                    "description": result.get("description", ""),
                }

        except FileNotFoundError:
            raise AIClientException(f"Image file not found: {image_path}")
        except Exception as e:
            raise AIClientException(f"StudioLM classification error: {str(e)}")
