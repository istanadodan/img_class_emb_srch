"""StudioLM AI Client - Local StudioLM API를 통한 이미지 분류"""

import base64
from openai import AsyncOpenAI
from openai.types.chat.chat_completion import ChatCompletion
from typing import Optional, Any, cast
from backend.clients.base import AIClient
from backend.system.config import settings
from backend.system.exceptions import AIClientException
from shared.classification_schema import ClassificationResult
from functools import lru_cache
from backend.system.config import settings


class StudioLMClient(AIClient):
    """StudioLM API 클라이언트"""

    def __init__(
        self,
        api_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: int = 180,
    ):
        self.api_url = api_url or settings.studiolm_api_url
        self.api_key = api_key or settings.studiolm_api_key
        self.llm_model = settings.llm_model_name
        self.embed_model = settings.embedding_model_name

        self.client = self.get_client(self.api_url + "/v1", self.api_key, timeout=timeout)

    @property
    def name(self) -> str:
        return "StudioLM"

    def get_model(self) -> str:
        """사용 중인 LLM 모델 이름 반환"""
        return self.llm_model

    def get_embd_model(self) -> str:
        """사용 중인 임데딩 모델 이름 반환"""
        return self.embed_model

    @lru_cache
    def get_client(self, api_url: str, api_key: str, timeout: int) -> AsyncOpenAI:

        client = AsyncOpenAI(
            base_url=api_url,
            api_key=api_key,
            timeout=timeout,
        )
        return client

    async def create_embeddings(self, input_text: str) -> list[float]:
        result = await self.client.embeddings.create(
            model=settings.embedding_model_name,
            input=input_text,
            extra_headers={
                "Content-Type": "application/json",
            },
        )
        return result.data[0].embedding

    async def generate_message(
        self, messages: list[Any], output_schema: Optional[dict] = None
    ) -> ChatCompletion:
        """메시지 생성"""
        return await self.client.chat.completions.create(
            model=self.llm_model,
            messages=messages,
            temperature=0.1,
            max_tokens=800,
            response_format=cast(Any, output_schema),
            extra_body={
                "enable_thinking": True,
                "thinking_budget": 600,
            },
            extra_headers={
                "Content-Type": "application/json",
            },
        )

    def is_available(self) -> bool:
        """StudioLM API 서버 연결 가능 여부 확인"""
        try:
            # 동기 헬스 체크는 유지하거나 비동기로 변경 가능
            import requests

            response = requests.get(f"{self.api_url}/health", timeout=5)
            return response.status_code == 200
        except Exception:
            return False

    async def classify_image(self, image_data: bytes) -> ClassificationResult:
        """
        이미지를 StudioLM API로 분류

        Args:
            image_path: 이미지 파일 경로
            image_data: 이미 로드된 이미지 객체 (선택 사항)

        Returns:
            분류 결과 (ClassificationResult 모델)
        """
        # [설정] 프롬프트 (생략...)
        SYSTEM_PROMPT = """You are a helpful assistant. Analyze the provided image and generate a response.
        Do not re-evaluate.
        No step-by-step reasoning.
        Before answering, reason briefly (under 300 words internally).
        Prioritize output quality over exhaustive reasoning.
        """
        USER_PROMPT = """/no_think

You are an image classifier. Your job is to output JSON instantly without re-reading or second-guessing.

## HARD RULES — apply in order, stop immediately at first match

RULE 1: If image contains a greeting message OR a year/date OR a holiday wish → category = "events"
RULE 2: If humans are the main subject → category = "people"
RULE 3: If animals are the main subject → category = "animal"
RULE 4: If flowers are the main subject → category = "flower"
RULE 5: If a specific location/building is the main subject → category = "place"
RULE 6: If nature/landscape is the main subject → category = "nature"
RULE 7: Everything else → category = "text"

## CRITICAL INSTRUCTIONS
- Apply RULE 1 first, always. It overrides all other rules.
- Do NOT analyze all rules. Stop at the first match.
- Do NOT write any reasoning, explanation, or thought process.
- Output ONLY the JSON object below. Nothing before or after it.

## Output
{"category": "...", "confidence": 0.0, "description": "판단 근거...", "objects": ["여성","낙타","..."]}
"""
        try:
            # API 전송을 위한 일시적 바이트 변환 및 Base64 인코딩
            image_base64 = base64.b64encode(image_data).decode("utf-8")

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": USER_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                        },
                    ],
                },
            ]

            # 비동기 호출
            completion = await self.generate_message(
                messages=messages,
                output_schema={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "classification_response",
                        "strict": True,
                        "schema": ClassificationResult.model_json_schema(),
                    },
                },
            )

            category_result = (
                completion.choices[0].message.content
                or cast(dict, completion.choices[0].message.model_extra)["reasoning_content"]
            )
            if not category_result:
                raise AIClientException("StudioLM API returned empty result")

            return ClassificationResult.model_validate_json(category_result)

        except Exception as e:
            raise AIClientException(f"StudioLM classification error: {str(e)}")

    async def get_image_data(self, image_path: Any):
        return
