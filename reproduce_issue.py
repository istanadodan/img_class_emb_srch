
import asyncio
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
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
    """Search images in DB."""
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
            "당신은 이미지 검색 도우미입니다. 다음 순서를 엄격히 지키세요:\n"
            "1. 사용자의 질문을 분석하여 'search_images_tool' 도구를 호출해 이미지를 검색하세요.\n"
            "2. 검색 도구의 결과를 확인한 후, 그 정보를 바탕으로 최종 응답을 SearchResponse 형식에 맞춰 작성하세요.\n"
            "도구를 사용하기 전에는 절대로 최종 SearchResponse 형식을 출력하지 마세요."
        ),
        response_format=SearchResponse,
    )

    inputs = {"messages": [HumanMessage(content="고양이 사진 찾아줘")]}
    print("Invoking agent...")
    result = await agent.ainvoke(inputs)
    print("\nResult type:", type(result))
    print("Result keys:", result.keys())
    if "structured_response" in result:
        print("Structured response:", result["structured_response"])
    
    for i, msg in enumerate(result.get("messages", [])):
        print(f"Message {i} ({type(msg).__name__}): {msg.content}")
        if hasattr(msg, "tool_calls"):
            print(f"  Tool calls: {msg.tool_calls}")

if __name__ == "__main__":
    asyncio.run(test_agent())
