
import asyncio
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain.tools import tool
from pydantic import SecretStr
import os
from dotenv import load_dotenv

load_dotenv()

@tool
def search_images_tool(query: str) -> str:
    """Search images."""
    return "result"

async def test_llm():
    llm = ChatOpenAI(
        base_url=os.getenv("STUDIOLM_API_URL") + "/v1",
        api_key=SecretStr(os.getenv("STUDIOLM_API_KEY")),
        model=os.getenv("LLM_MODEL_NAME"),
        temperature=0,
    )
    
    llm_with_tools = llm.bind_tools([search_images_tool])
    
    print("Calling LLM with tools...")
    res = await llm_with_tools.ainvoke([HumanMessage(content="고양이 사진 찾아줘")])
    print(f"Content: {res.content}")
    print(f"Tool calls: {res.tool_calls}")

if __name__ == "__main__":
    asyncio.run(test_llm())
