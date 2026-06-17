
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

async def test_agent():
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
            "2. 검색된 정보를 바탕으로 최종 응답을 SearchResponse 형식에 맞춰 작성하세요.\n"
            "3. 검색 결과가 없다면 빈 리스트를 반환하세요.\n"
            "반드시 도구를 먼저 사용해야 함을 명심하세요."
        ),
        response_format=SearchResponse,
    )

    inputs = {"messages": [HumanMessage(content="고양이 사진 찾아줘")]}
    print("Invoking agent...")
    result = await agent.ainvoke(inputs)
    
    print("\n--- Result Analysis ---")
    print(f"Keys in result: {result.keys()}")
    
    if "structured_response" in result:
        print(f"Structured Response: {result['structured_response']}")
    else:
        print("Structured Response is MISSING in result keys.")

    messages = result.get("messages", [])
    for i, msg in enumerate(messages):
        print(f"Message {i} ({type(msg).__name__}): {msg.content}")
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            print(f"  Tool calls: {msg.tool_calls}")

if __name__ == "__main__":
    asyncio.run(test_agent())
