"""Image Classification Workflow - LangGraph 기반 이미지 분류 워크플로우"""

from typing import TypedDict, Optional
from langgraph.graph import StateGraph, START, END
from backend.clients.base import AIClient


class ClassificationState(TypedDict):
    """이미지 분류 워크플로우 상태"""

    image_path: str
    category: Optional[str]
    confidence: float
    description: str
    error: Optional[str]


def classify_node(state: ClassificationState, ai_client: AIClient) -> ClassificationState:
    """
    이미지 분류 노드

    Args:
        state: 현재 상태
        ai_client: AI 클라이언트
    """
    try:
        # 동기 래퍼 필요 (실제로는 비동기 처리)
        import asyncio

        result = asyncio.run(ai_client.classify_image(state["image_path"]))

        state["category"] = result["category"]
        state["confidence"] = result["confidence"]
        state["description"] = result.get("description", "")
        state["error"] = None

    except Exception as e:
        state["error"] = str(e)
        state["category"] = None
        state["confidence"] = 0.0

    return state


def validate_result_node(state: ClassificationState) -> ClassificationState:
    """
    분류 결과 검증 노드

    Args:
        state: 현재 상태
    """
    if state["error"]:
        return state

    # 신뢰도 임계값 체크
    if state["confidence"] < 0.5:
        state["error"] = f"Low confidence score: {state['confidence']}"

    return state


def create_classification_workflow(ai_client: AIClient) -> StateGraph:
    """
    이미지 분류 워크플로우 생성

    Args:
        ai_client: 사용할 AI 클라이언트

    Returns:
        컴파일된 워크플로우
    """
    workflow = StateGraph(ClassificationState)

    # 노드 추가
    workflow.add_node(
        "classify", lambda state: classify_node(state, ai_client)
    )
    workflow.add_node("validate", validate_result_node)

    # 엣지 연결
    workflow.add_edge(START, "classify")
    workflow.add_edge("classify", "validate")
    workflow.add_edge("validate", END)

    return workflow.compile()
