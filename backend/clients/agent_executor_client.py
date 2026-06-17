"""Agent Executor Client - LangChain 1.3.4 (LangGraph 기반) 도구 사용 에이전트"""

from typing import List, Any, Optional, Dict, Union
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph.state import CompiledStateGraph
from pydantic import SecretStr

from backend.system.config import settings
from backend.system.utils import parse_structured_response
from backend.service.tools import PICRECOG_TOOLS
from shared.search_schema import SearchResponse  # Import the schema


class AgentExecutorClient:
    """
    LangChain 1.3.4 버전에 최적화된 에이전트 클라이언트.
    create_agent를 통해 생성된 LangGraph 기반 워크플로우를 사용하여 도구를 실행합니다.
    """

    def __init__(self, tools: Optional[List[Any]] = None):
        # 1. LLM 초기화 (Pylance 타입 힌트 적용)
        self.llm: BaseChatModel = ChatOpenAI(
            base_url=settings.studiolm_api_url + "/v1",
            api_key=SecretStr(settings.studiolm_api_key),
            model=settings.llm_model_name,
            temperature=0.1,
            streaming=True,
        )

        # 2. 에이전트 그래프 생성 (response_format 제거 - 도구 호출 호환성 문제)
        self.agent: CompiledStateGraph = create_agent(
            model=self.llm,
            tools=tools or [],
            system_prompt=(
                "당신은 이미지 검색 도우미입니다. 사용자의 질의를 분석하여 가장 적합한 이미지를 찾아주는 역할을 수행합니다.\n\n"
                "### 작업 지침:\n"
                "1. **도구 호출**: 먼저 가용한 도구를 호출하여 관련 이미지를 검색하세요. 충분한 후보군을 확보하기 위해 필요하다면 검색어를 구체화하여 호출하세요.\n"
                "2. **검증 및 필터링 (중요)**: 도구가 반환한 각 이미지의 '설명' 필드를 사용자의 질의와 대조하여 엄격하게 필터링하세요.\n"
                "   - **수량 일치**: 예를 들어 '여성 2명'을 요청했다면, 설명에 명시적으로 인원수가 일치하거나 수량이 부합하는 이미지만 선택하세요.\n"
                "   - **속성 일치**: 색상, 사물, 행위 등이 사용자의 요청과 정확히 일치하는지 확인하세요.\n"
                "3. **결과 작성**: 필터링을 통과한 이미지들만 사용하여 최종 응답을 다음 JSON 형식으로 작성하세요. 각 이미지마다 **'justification'** 필드에 해당 이미지가 왜 선택되었는지, 질의와 어떤 점이 일치하는지에 대한 **개별적인 분석 근거**를 상세히 작성하세요. **'similarity'**는 반드시 0.0에서 1.0 사이의 소수로 작성하세요:\n"
                "{\n"
                "  \"results\": [\n"
                "    {\n"
                "      \"image_id\": 1, \n"
                "      \"similarity\": 0.95, \n"
                "      \"path\": \"/path/to/image.jpg\",\n"
                "      \"justification\": \"이 이미지는 파란색 옷을 입은 여성이 2명 포함되어 있어 질의와 정확히 일치합니다.\"\n"
                "    }\n"
                "  ]\n"
                "}\n"

                '4. **결과 없음**: 필터링 후 조건에 맞는 이미지가 하나도 없다면 {"results": []}를 반환하세요.\n'
                "5. **출력**: 부가 설명 없이 반드시 JSON 형식만 출력하세요."
            ),
        )

    async def ask(self, query: str) -> List[Dict[str, Any]]:
        """
        사용자 질의를 처리하고 구조화된 검색 결과 목록을 반환합니다.
        """
        # 1. 입력 메시지 구성
        inputs: Dict[str, List[BaseMessage]] = {"messages": [HumanMessage(content=query)]}

        # 2. 에이전트 실행
        result: Dict[str, Any] = await self.agent.ainvoke(inputs)

        # 3. 구조화된 결과 추출 (마지막 메시지에서 파싱)
        last_msg = result.get("messages", [None])[-1]
        search_res = parse_structured_response(last_msg, SearchResponse)

        if search_res and isinstance(search_res, SearchResponse):
            return [res.model_dump() for res in search_res.results]

        return []


def get_agent_executor_client() -> AgentExecutorClient:
    """도구가 바인딩된 AgentExecutorClient 인스턴스를 반환하는 팩토리 함수"""
    return AgentExecutorClient(tools=PICRECOG_TOOLS)
