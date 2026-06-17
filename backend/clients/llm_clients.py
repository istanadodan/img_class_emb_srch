"""StudioLM AI Client - Local StudioLM API 연동 (순수 LLM 및 Embedding)"""

import base64
from openai import AsyncOpenAI
from openai.types.chat.chat_completion import ChatCompletion
from typing import Optional, Any, cast, List
from backend.clients.base import AIClient
from backend.system.config import settings
from backend.system.exceptions import AIClientException
from shared.classification_schema import ClassificationResult
from functools import lru_cache


class StudioLLMClient(AIClient):
    """StudioLLM API 클라이언트 (채팅/분류 전용)"""

    def __init__(
        self, api_url: Optional[str] = None, api_key: Optional[str] = None, timeout: int = 180
    ):
        self.api_url = api_url or settings.studiolm_api_url
        self.api_key = api_key or settings.studiolm_api_key
        self.llm_model = settings.llm_model_name
        self.client = self.get_client(self.api_url + "/v1", self.api_key, timeout=timeout)

    @property
    def name(self) -> str:
        return "StudioLLM"  # Name reflects new focus

    def get_model(self) -> str:
        """사용 중인 LLM 모델 이름 반환"""
        return self.llm_model

    # Removed get_embd_model and create_embeddings as they belong to StudioEmbeddingClient

    @lru_cache
    def get_client(self, api_url: str, api_key: str, timeout: int) -> AsyncOpenAI:
        return AsyncOpenAI(base_url=api_url, api_key=api_key, timeout=timeout)

    # create_embeddings method removed

    async def generate_message(
        self, messages: List[Any], output_schema: Optional[dict] = None
    ) -> ChatCompletion:
        return await self.client.chat.completions.create(
            model=self.llm_model,
            messages=messages,
            temperature=0.1,
            max_tokens=800,
            response_format=cast(Any, output_schema),
            extra_body={"enable_thinking": True, "thinking_budget": 600},
            extra_headers={"Content-Type": "application/json"},
        )

    async def classify_image(self, image_data: bytes) -> ClassificationResult:
        try:
            image_base64 = base64.b64encode(image_data).decode("utf-8")
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Classify this image..."},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                        },
                    ],
                },
            ]
            completion = await self.generate_message(
                messages=messages,
                output_schema={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "res",
                        "strict": True,
                        "schema": ClassificationResult.model_json_schema(),
                    },
                },
            )
            return ClassificationResult.model_validate(completion.choices[0].message.content)
        except Exception as e:
            raise AIClientException(f"Classification error: {str(e)}")


def get_studio_LLM_client() -> StudioLLMClient:
    """StudioEmbeddingClient 인스턴스를 반환하는 팩토리 함수"""
    return StudioLLMClient()
