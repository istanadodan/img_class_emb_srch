"""System Utilities - 공통 유틸리티 함수 모음"""

import json
import re
from typing import Any, Optional, Type, TypeVar
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def parse_structured_response(response: Any, model_class: Type[T]) -> Optional[T]:
    """
    LLM/에이전트의 응답에서 특정 Pydantic 모델 형식의 구조화된 데이터를 추출하고 파싱합니다.

    Args:
        response: AIMessage, str, 또는 dict 형식의 응답 객체
        model_class: 파싱할 Pydantic 모델 클래스

    Returns:
        파싱된 모델 인스턴스 또는 실패 시 None
    """
    if response is None:
        return None

    content = ""
    # 1. 콘텐츠 추출
    if hasattr(response, "content") and response.content:
        content = response.content
    elif hasattr(response, "model_extra") and isinstance(response.model_extra, dict):
        content = response.model_extra["reasoning_content"]
    elif isinstance(response, str):
        content = response
    elif isinstance(response, dict):
        try:
            return model_class.model_validate(response)
        except Exception:
            return None

    if not content or not isinstance(content, str):
        return None

    # 2. JSON 블록 또는 객체 추출 시도
    try:
        # ```json ... ``` 블록 찾기
        json_match = re.search(r"```json\s*(\{.*?\})\s*```", content, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # 첫 번째 { 와 마지막 } 사이 추출
            start = content.find("{")
            end = content.rfind("}")
            if start != -1 and end != -1:
                json_str = content[start : end + 1]
            else:
                json_str = content

        return model_class.model_validate_json(json_str)
    except Exception:
        return None
