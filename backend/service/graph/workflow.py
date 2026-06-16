"""Image Classification Workflow - LangGraph 기반 이미지 분류 워크플로우"""

import asyncio
from typing import Any, TypedDict, Optional, Annotated, Literal, cast
from langgraph.graph import StateGraph, START, END, state
from backend.clients.base import AIClient
import logging
import operator
from datetime import datetime
from backend.system.exceptions import AIClientException
from shared.classification_schema import ClassificationResult
from backend.service.vector_search import VectorSearchService

from backend.system.event_handler import UIEventHandler, WorkflowLogger

logger = logging.getLogger(__name__)


class ClassificationState(TypedDict):
    """이미지 분류 워크플로우 상태"""

    image_path: str
    image_bytes: bytes
    embedding: Optional[list[float]]
    category: Optional[str]
    confidence: float
    description: str
    objects: list[str]
    error: Optional[str]
    retry_prompt: Optional[str]
    retry_cnt: int
    max_retires: int
    logs: Annotated[list[dict], operator.add]


async def classify_node(
    state: ClassificationState, ai_client: AIClient, event_queue: Optional[asyncio.Queue] = None
) -> ClassificationState:
    """
    이미지 분류 노드
    """
    node_name = "classify"
    await UIEventHandler.publish(
        event_queue, "progress", "이미지 분류 분석 중...", path=state["image_path"], node=node_name
    )

    try:
        # Service에서 미리 로드한 image_obj를 넘겨 중복 로드 방지
        result: ClassificationResult = await ai_client.classify_image(
            image_data=state.get("image_bytes")
        )

        state.update(cast(ClassificationState, result.model_dump(include=set(state))))
        state["error"] = None

        status = "completed"
        message = f"Classification successful: {state['category']} ({state['confidence']:.2f})"

    except AIClientException as e:
        state["error"] = str(e)
        if not state.get("category"):
            state["category"] = "no-class"
        status = "failed"
        message = f"Classification error: {state['error']}"
        await UIEventHandler.publish(
            event_queue, "error", message, path=state["image_path"], node=node_name
        )

    WorkflowLogger.log(node_name, message, level="error" if status == "failed" else "info")
    state["logs"] = [
        WorkflowLogger.make_state_log(
            node_name, message, status=status, category=state.get("category")
        )
    ]
    return state


async def validate_result_node(
    state: ClassificationState, event_queue: Optional[asyncio.Queue] = None
) -> ClassificationState:
    """
    분류 결과 검증 노드
    """
    node_name = "validate"
    await UIEventHandler.publish(
        event_queue, "progress", "분류 결과 검증 중...", path=state["image_path"], node=node_name
    )

    # 1. API 에러가 발생한 경우 (신뢰도 부족 제외)
    if state.get("error"):
        state["retry_cnt"] += 1
        status = "failed"
        message = f"Retrying due to API error (Attempt {state['retry_cnt']}/{state['max_retires']})"
    # 2. 신뢰도가 낮은 경우
    elif state["confidence"] < 0.5:
        state["error"] = f"Low confidence score: {state['confidence']:.2f}"
        state["retry_cnt"] += 1
        status = "failed"
        message = f"Retrying due to low confidence: {state['confidence']:.2f} (Attempt {state['retry_cnt']}/{state['max_retires']})"
    # 3. 검증 통과
    else:
        status = "completed"
        message = "Validation passed"

    WorkflowLogger.log(node_name, message)
    state["logs"] = [
        WorkflowLogger.make_state_log(
            node_name, message, status=status, confidence=state["confidence"]
        )
    ]
    return state


async def embedding_node(
    state: ClassificationState, embedding_client: Any, event_queue: Optional[asyncio.Queue] = None
) -> ClassificationState:
    """
    이미지 임베딩 생성 노드
    """
    node_name = "embedding"
    await UIEventHandler.publish(
        event_queue, "progress", "이미지 벡터 추출 중...", path=state["image_path"], node=node_name
    )
    try:
        # 모델 추론(CPU/GPU)은 별도 스레드에서 실행하여 이벤트 루프 차단 방지
        embedding_vector = await embedding_client.embed_image(state["image_bytes"])
        state["embedding"] = embedding_vector
        state["error"] = None
        status = "completed"
        message = f"Embedding generated successfully (Dim: {len(embedding_vector)})"
    except Exception as e:
        state["error"] = f"Embedding generation failed: {str(e)}"
        status = "failed"
        message = state["error"]
        await UIEventHandler.publish(
            event_queue, "error", message, path=state["image_path"], node=node_name
        )

    WorkflowLogger.log(node_name, message, level="error" if status == "failed" else "info")
    state["logs"] = [WorkflowLogger.make_state_log(node_name, message, status=status)]
    return state


async def vector_store_node(
    state: ClassificationState, embedding_client: Any, event_queue: Optional[asyncio.Queue] = None
) -> ClassificationState:
    """이미지 벡터 저장 노드 (no-class 포함 모든 결과 저장)"""
    node_name = "store"
    category = state.get("category") or "no-class"
    await UIEventHandler.publish(
        event_queue,
        "progress",
        f"DB 저장 중 ({category})...",
        path=state["image_path"],
        node=node_name,
    )

    try:
        service = VectorSearchService(embedding_client=embedding_client)
        embedding = state.get("embedding")
        await service.save_embedding(
            image_path=state["image_path"],
            embedding=embedding if embedding is not None else [],
            category=category,
            confidence=state.get("confidence", 0.0),
            description=state.get("description", "Auto-processed"),
            objects=state.get("objects", []),
        )
        status = "completed"
        message = f"Successfully stored in database as [{category}]"

        # UI에 실시간 성공 결과 전송 (실시간 진행률 반영)
        await UIEventHandler.publish(
            event_queue,
            "result",
            f"분류 완료: {category}",
            path=state["image_path"],
            category=category,
            confidence=state.get("confidence", 0.0),
        )
    except Exception as e:
        state["error"] = f"Storage failed: {str(e)}"
        status = "failed"
        message = state["error"]
        await UIEventHandler.publish(
            event_queue, "error", message, path=state["image_path"], node=node_name
        )

    WorkflowLogger.log(node_name, message, level="error" if status == "failed" else "info")
    state["logs"] = [
        WorkflowLogger.make_state_log(node_name, message, status=status, category=category)
    ]
    return state


def create_classification_workflow(
    ai_client: AIClient, embedding_client: Any = None
) -> state.CompiledStateGraph:
    """
    이미지 분류 워크플로우 생성
    """
    from langchain_core.runnables import RunnableConfig

    async def classify_wrapper(
        state: ClassificationState, config: RunnableConfig
    ) -> ClassificationState:
        q = config.get("configurable", {}).get("event_queue")
        return await classify_node(state, ai_client, event_queue=q)

    async def validate_wrapper(
        state: ClassificationState, config: RunnableConfig
    ) -> ClassificationState:
        q = config.get("configurable", {}).get("event_queue")
        return await validate_result_node(state, event_queue=q)

    async def embedding_wrapper(
        state: ClassificationState, config: RunnableConfig
    ) -> ClassificationState:
        q = config.get("configurable", {}).get("event_queue")
        return await embedding_node(state, embedding_client, event_queue=q)

    async def vector_store_wrapper(
        state: ClassificationState, config: RunnableConfig
    ) -> ClassificationState:
        q = config.get("configurable", {}).get("event_queue")
        return await vector_store_node(state, embedding_client, event_queue=q)

    workflow = StateGraph(state_schema=ClassificationState)

    # 노드 추가
    workflow.add_node("classify", classify_wrapper)
    workflow.add_node("validate", validate_wrapper)
    workflow.add_node("embedding", embedding_wrapper)
    workflow.add_node("store", vector_store_wrapper)

    # 엣지 연결
    workflow.add_edge(START, "classify")
    workflow.add_edge("classify", "validate")

    # 1. 분류 후 라우팅
    workflow.add_conditional_edges(
        "validate",
        route_after_classify,
        {"retry": "classify", "continue": "embedding"},
    )

    # 2. 임베딩 후 라우팅
    workflow.add_conditional_edges(
        "embedding",
        route_after_embedding,
        {"store": "store", "fail": END},
    )
    workflow.add_edge("store", END)

    return workflow.compile()


def route_after_classify(state: ClassificationState) -> Literal["retry", "continue"]:
    """분류 검증 후 라우팅"""
    retry_cnt = state.get("retry_cnt", 0)
    max_retires = state.get("max_retires", 3)

    # 성공했거나 최대 재시도 도달 시 임베딩으로 진행
    if not state.get("error") and state.get("category") not in [None, "no-class"]:
        return "continue"

    if retry_cnt >= max_retires:
        # 최대 재시도 도달 시에도 임베딩으로 진행 (no-class 상태)
        if not state.get("category"):
            state["category"] = "no-class"
        return "continue"

    return "retry"


def route_after_embedding(state: ClassificationState) -> Literal["store", "fail"]:
    """임베딩 후 저장 여부 결정"""
    # 임베딩 벡터가 없거나 에러가 있으면 저장하지 않고 종료
    if state.get("error") or state.get("embedding") is None:
        logger.error(f"Embedding failed for {state['image_path']}. Skipping storage.")
        return "fail"

    return "store"


"""유틸리티"""


def make_log(node: str, message: str, **extra) -> dict:
    return dict(
        node=node, message=message, timestamp=datetime.now().isoformat(timespec="seconds"), **extra
    )
