"""Embedding Client - LM Studio를 통한 벡터 임베딩 생성"""

import numpy as np
import httpx
from typing import Optional, cast
from backend.system.config import settings
from backend.system.exceptions import AIClientException
from typing import Any
import numpy as np
from PIL import Image
from io import BytesIO
from base64 import b64encode
from backend.clients.llm_clients import StudioLMClient
from shared.classification_schema import ClassificationResult


class EmbeddingClient:
    """LM Studio 임베딩 클라이언트"""

    def __init__(
        self,
        api_url: Optional[str] = None,
        timeout: int = 120,
    ):
        from openai import AsyncOpenAI

        self.api_url = api_url or settings.studiolm_api_url
        self.timeout = timeout
        self.lmstudio_client = StudioLMClient()

    def get_model(self) -> str:
        """사용 중인 임베딩 모델 이름 반환"""
        return self.lmstudio_client.get_embd_model()

    # 추가 기능: 얼굴 인식 및 임베딩 생성
    async def embed_image(self, image_data: bytes) -> np.ndarray:
        """
        이미지 내 모든 얼굴에 대해 임베딩 생성

        Args:
            image_path: 이미지 파일 경로

        Returns:
            얼굴 임베딩 정보 리스트
        """
        try:
            # 1. 이미지 로드 및 Base64 인코딩
            # with open(image_path, "rb") as f:
            #     image_data = base64.b64encode(f.read()).decode("utf-8")
            image_base64 = b64encode(image_data).decode("utf-8")
            # 2. 이미지 속 얼굴 인식 (여기서는 예시로 전체 이미지 사용)
            face_descriptions = await self._describe_image(image_base64)

            # 3. 텍스트를 벡터로 변환
            face_embedding = await self._text_to_vector(face_descriptions)

            print(f"face decriptions: {face_descriptions}")
            print(f"Generated embedding shape: {np.array(face_embedding).shape}")
            return face_embedding.tolist()

        except Exception as e:
            raise AIClientException(f"Face embedding generation error: {str(e)}")

    # 내부 기능 1) 이미지 -> 표현글
    async def _describe_image(self, image_data: str) -> str:

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "이미지의 주요 인물만 대상으로 하며 배경 인물은 완전히 무시할 것. "
                            "각 인물의 얼굴을 이미지 검색 및 인물 식별에 활용할 수 있도록 매우 상세하고 객관적으로 묘사할 것. "
                            "추측이나 해석은 금지하며 실제로 보이는 특징만 기술할 것. "
                            "다음 항목을 반드시 포함할 것:\n"
                            "1. 성별 추정 및 나이대\n"
                            "2. 얼굴형 (둥근형, 계란형, 역삼각형, 사각형, 긴 얼굴형 등)\n"
                            "3. 얼굴 비율 및 특징 (광대, 턱선, 턱끝, 이마 넓이, 얼굴 길이 등)\n"
                            "4. 피부톤 및 피부 특징 (밝기, 주근깨, 점, 잡티, 주름, 수염 여부 등)\n"
                            "5. 눈 특징 (크기, 형태, 쌍꺼풀 여부, 눈꼬리 방향, 눈썹과의 거리, 눈동자 색상 등)\n"
                            "6. 눈썹 특징 (굵기, 길이, 밀도, 각도, 모양)\n"
                            "7. 코 특징 (높이, 폭, 콧대 형태, 코끝 모양, 콧볼 크기)\n"
                            "8. 입 특징 (입술 두께, 입꼬리 방향, 인중 길이, 치아 노출 여부)\n"
                            "9. 귀 특징 (노출 여부, 크기, 형태)\n"
                            "10. 헤어스타일 (길이, 가르마 방향, 컬 여부, 앞머리, 머리색)\n"
                            "11. 안경 및 액세서리 (안경 종류, 선글라스, 귀걸이, 피어싱 등)\n"
                            "12. 표정 (웃음, 무표정, 찡그림, 입 벌림 등)\n"
                            "13. 얼굴 방향 (정면, 좌측, 우측, 위, 아래)\n"
                            "14. 촬영 각도 (정면샷, 측면샷, 로우앵글, 하이앵글)\n"
                            "15. 식별 가능한 고유 특징 (점, 흉터, 비대칭, 독특한 헤어라인, 특이한 얼굴 구조 등)\n"
                            "최종 결과는 인물별로 구조화하여 작성할 것. "
                            "특히 다른 사람과 구별 가능한 특징을 우선적으로 자세히 기술할 것."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                    },
                ],
            },
        ]

        completion = await self.lmstudio_client.generate_message(
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
        # async with httpx.AsyncClient(timeout=self.timeout) as client:
        #     response = await client.post(
        #         f"{self.api_url}/v1/chat/completions",
        #         json={
        #             "model": "Qwen/Qwen3.5-9B",
        #             "messages": []
        #         },
        #         headers={"Content-Type": "application/json"},
        #     )
        # response.raise_for_status()
        return (
            completion.choices[0].message.content
            or cast(dict, completion.choices[0].message.model_extra)["reasoning_content"]
        )

    # 2) 표현글 -> embedding
    async def _text_to_vector(self, text: str) -> np.ndarray:
        result: list[float] = await self.lmstudio_client.create_embeddings(text)

        # async with httpx.AsyncClient(timeout=self.timeout) as client:
        #     response = await client.post(
        #         f"{self.api_url}/v1/embeddings",
        #         json={
        #             "model": "text-embedding-nomic-embed-text-v1.5@q8_0",
        #             "input": text,
        #         },
        #         headers={"Content-Type": "application/json"},
        #     )
        # response.raise_for_status()
        # result = response.json()
        # vector = np.array(response.json()["data"][0]["embedding"], dtype=np.float32)
        embeddings = np.array(result, dtype=np.float32)
        # L2 정규형
        norm = np.linalg.norm(embeddings)
        return embeddings / norm if norm > 0 else embeddings


def get_emb_client():
    return EmbeddingClient()
