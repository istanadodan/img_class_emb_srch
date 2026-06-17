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
    node_name: str = "classify"
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

        status: str = "completed"
        message: str = f"Classification successful: {state['category']} ({state['confidence']:.2f})"

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
    node_name: str = "validate"
    await UIEventHandler.publish(
        event_queue, "progress", "분류 결과 검증 중...", path=state["image_path"], node=node_name
    )

    # 1. API 에러가 발생한 경우 (신뢰도 부족 제외)
    status: str
    message: str
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
    이미지 텍스트 묘사 기반 임베딩 생성 노드
    """
    node_name: str = "embedding"
    await UIEventHandler.publish(
        event_queue, "progress", "이미지 특징 벡터 추출 중...", path=state["image_path"], node=node_name
    )
    try:
        # 1. 텍스트 기반 특징 조합 (카테고리 + 상세 설명 + 객체 태그)
        # LLM이 생성한 풍부한 묘사를 바탕으로 시맨틱 벡터 생성
        features_text: str = f"Category: {state.get('category')}. Description: {state.get('description')}. Tags: {', '.join(state.get('objects', []))}"
        
        # 2. 텍스트 임베딩 생성 (np.ndarray -> list[float] 변환)
        embedding_vector: Any = await embedding_client.create_embeddings(features_text)
        state["embedding"] = embedding_vector.tolist() if hasattr(embedding_vector, "tolist") else list(embedding_vector)
        
        state["error"] = None
        status: str = "completed"
        message: str = f"Text-based embedding generated successfully (Dim: {len(state['embedding'])})"
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
    node_name: str = "store"
    category: str = state.get("category") or "no-class"
    await UIEventHandler.publish(
        event_queue,
        "progress",
        f"DB 저장 중 ({category})...",
        path=state["image_path"],
        node=node_name,
    )

    try:
        service: VectorSearchService = VectorSearchService(embedding_client=embedding_client)
        embedding: Optional[list[float]] = state.get("embedding")
        await service.save_embedding(
            image_path=state["image_path"],
            embedding=embedding if embedding is not None else [],
            category=category,
            confidence=state.get("confidence", 0.0),
            description=state.get("description", "Auto-processed"),
            objects=state.get("objects", []),
        )
        status: str = "completed"
        message: str = f"Successfully stored in database as [{category}]"

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


async def error_handler_node(
    state: ClassificationState, event_queue: Optional[asyncio.Queue] = None
) -> ClassificationState:
    """
    워크플로우 오류 처리 및 UI 보고 노드
    """
    node_name: str = "error_handler"
    error_msg: str = state.get("error") or "Unknown error occurred during processing"
    
    # UI에 에러 이벤트 발행 (처리 개수 반영 및 로그 출력용)
    await UIEventHandler.publish(
        event_queue, 
        "error", 
        f"처리 실패: {error_msg}", 
        path=state["image_path"], 
        node=node_name
    )

    WorkflowLogger.log(node_name, f"Workflow failed for {state['image_path']}: {error_msg}", level="error")
    state["logs"] = [
        WorkflowLogger.make_state_log(node_name, error_msg, status="failed")
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

    async def error_handler_wrapper(
        state: ClassificationState, config: RunnableConfig
    ) -> ClassificationState:
        q = config.get("configurable", {}).get("event_queue")
        return await error_handler_node(state, event_queue=q)

    workflow = StateGraph(state_schema=ClassificationState)

    # 노드 추가
    workflow.add_node("classify", classify_wrapper)
    workflow.add_node("validate", validate_wrapper)
    workflow.add_node("embedding", embedding_wrapper)
    workflow.add_node("store", vector_store_wrapper)
    workflow.add_node("error_handler", error_handler_wrapper)

    # 엣지 연결
    workflow.add_edge(START, "classify")
    workflow.add_edge("classify", "validate")

    # 1. 분류 후 라우팅
    workflow.add_conditional_edges(
        "validate",
        route_after_classify,
        {"retry": "classify", "continue": "embedding", "fail": "error_handler"},
    )

    # 2. 임베딩 후 라우팅
    workflow.add_conditional_edges(
        "embedding",
        route_after_embedding,
        {"store": "store", "fail": "error_handler"},
    )
    
    # 최종 종료 연결
    workflow.add_edge("store", END)
    workflow.add_edge("error_handler", END)

    return workflow.compile()


def route_after_classify(state: ClassificationState) -> Literal["retry", "continue", "fail"]:
    """분류 검증 후 라우팅"""
    retry_cnt: int = state.get("retry_cnt", 0)
    max_retires: int = state.get("max_retires", 3)

    # 1. 성공 케이스 (에러 없고 카테고리 정상)
    if not state.get("error") and state.get("category") not in [None, "no-class"]:
        return "continue"

    # 2. 재시도 가능 여부 체크
    if retry_cnt < max_retires:
        return "retry"

    # 3. 최대 재시도 도달 시 (에러가 남아있는 경우)
    # 오류 처리 노드로 보내서 UI에 최종 실패 보고
    return "fail"


def route_after_embedding(state: ClassificationState) -> Literal["store", "fail"]:
    """임베딩 후 저장 여부 결정"""
    # 임베딩 벡터가 없거나 에러가 있으면 오류 처리 노드로 이동
    if state.get("error") or state.get("embedding") is None:
        return "fail"

    return "store"


"""유틸리티"""


def make_log(node: str, message: str, **extra) -> dict:
    return dict(
        node=node, message=message, timestamp=datetime.now().isoformat(timespec="seconds"), **extra
    )
