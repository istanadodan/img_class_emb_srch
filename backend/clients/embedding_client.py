"""Studio Embedding Client - LM Studio API를 통한 벡터 임베딩 생성 전용"""

import numpy as np
import httpx
from typing import Optional, Any, List
from backend.system.config import settings
from backend.system.exceptions import AIClientException
from functools import lru_cache
from openai import AsyncOpenAI


class StudioEmbeddingClient:
    """LM Studio 임베딩 클라이언트 (임베딩 전용)"""

    def __init__(
        self,
        api_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: int = 180,
    ):
        self.api_url = api_url or settings.studiolm_api_url
        self.api_key = api_key or settings.studiolm_api_key
        self.emb_model = settings.embedding_model_name
        self.emb_client = self._get_openai_client(
            self.api_url + "/v1", self.api_key, timeout=timeout
        )

    @lru_cache
    def _get_openai_client(self, api_url: str, api_key: str, timeout: int) -> AsyncOpenAI:
        """AsyncOpenAI 클라이언트 인스턴스를 캐싱하여 반환"""
        return AsyncOpenAI(
            base_url=api_url,
            api_key=api_key,
            timeout=timeout,
        )

    def get_model(self) -> str:
        """사용 중인 임베딩 모델 이름 반환"""
        return self.emb_model

    async def create_embeddings(self, input_text: str) -> np.ndarray:
        """
        텍스트의 임베딩 벡터를 생성

        Args:
            input_text: 임베딩할 텍스트

        Returns:
            임베딩 벡터 (np.ndarray)
        """
        try:
            result = await self.emb_client.embeddings.create(
                model=self.get_model(),
                input=input_text,
                extra_headers={"Content-Type": "application/json"},
            )
            embeddings = np.array(result.data[0].embedding, dtype=np.float32)
            # L2 정규화 (LM Studio Qwen 모델이 이미 정규화되어 있을 수 있으나, 안전을 위해 다시 적용)
            norm = np.linalg.norm(embeddings)
            return embeddings / norm if norm > 0 else embeddings
        except Exception as e:
            raise AIClientException(f"Embedding creation error: {str(e)}")


def get_studio_embedding_client() -> StudioEmbeddingClient:
    """StudioEmbeddingClient 인스턴스를 반환하는 팩토리 함수"""
    return StudioEmbeddingClient()
