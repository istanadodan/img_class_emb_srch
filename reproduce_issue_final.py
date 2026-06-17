
import asyncio
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, BaseMessage
from langchain.tools import tool
from pydantic import SecretStr
import os
from dotenv import load_dotenv
import json
import re

load_dotenv()

class ImageSearchResult(BaseModel):
    image_id: int
    similarity: float
    path: str

class SearchResponse(BaseModel):
    results: List[ImageSearchResult]

@tool
async def search_images_tool(query: str, category: Optional[str] = None) -> str:
    """PicRecog DB에서 자연어 질의를 바탕으로 유사한 이미지를 검색합니다."""
    print(f"\n>>> tool called with query: {query}, category: {category}")
    return "- ID: 1\n- 경로: /path/1.jpg\n  유사도: 95.0%\n  설명: A cute cat"

def parse_response(response: Any) -> Optional[SearchResponse]:
    if response is None: return None
    content = response.content if hasattr(response, "content") else (response if isinstance(response, str) else "")
    if not content: return None
    try:
        json_match = re.search(r"```json\s*(\{.*?\})\s*```", content, re.DOTALL)
        if json_match: content = json_match.group(1)
        else:
            start, end = content.find("{"), content.rfind("}")
            if start != -1 and end != -1: content = content[start : end + 1]
        return SearchResponse.model_validate_json(content)
    except Exception as e:
        print(f"Parse error: {e}")
        return None

async def test_agent_final():
    llm = ChatOpenAI(
        base_url=os.getenv("STUDIOLM_API_URL") + "/v1",
        api_key=SecretStr(os.getenv("STUDIOLM_API_KEY")),
        model=os.getenv("LLM_MODEL_NAME"),
        temperature=0.1,
    )

    agent = create_agent(
        model=llm,
        tools=[search_images_tool],
        system_prompt=(
            "당신은 이미지 검색 도우미입니다. 사용자의 질의에 따라 다음 단계를 수행하세요:\n"
            "1. 먼저 'search_images_tool'을 호출하여 관련 이미지를 검색하세요.\n"
            "2. 검색된 정보를 바탕으로 최종 응답을 반드시 다음 JSON 형식으로만 작성하세요:\n"
            "{\n"
            "  \"results\": [\n"
            "    {\"image_id\": 1, \"similarity\": 0.95, \"path\": \"/path/to/image.jpg\"}\n"
            "  ]\n"
            "}\n"
            "3. 검색 결과가 없다면 {\"results\": []}를 반환하세요.\n"
            "4. 반드시 도구를 먼저 사용해야 함을 명심하세요. 부가 설명 없이 JSON만 출력하세요."
        ),
    )

    inputs = {"messages": [HumanMessage(content="고양이 사진 찾아줘")]}
    print("Invoking agent...")
    result = await agent.ainvoke(inputs)
    
    last_msg = result.get("messages", [None])[-1]
    print(f"\nLast Message Content: {last_msg.content}")
    
    search_res = parse_response(last_msg)
    if search_res:
        print("\nParsed SearchResponse successfully!")
        print(search_res.model_dump_json(indent=2))
    else:
        print("\nFailed to parse SearchResponse.")

if __name__ == "__main__":
    asyncio.run(test_agent_final())
