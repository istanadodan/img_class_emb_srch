"""LangChain Tools for PicRecog - 비즈니스 로직 연동 도구 모음"""

from langchain.tools import tool
from backend.service.vector_search import VectorSearchService
from backend.clients.embedding_client import get_studio_embedding_client as emb_client
from typing import Optional


@tool
async def search_images_tool(query: str, category: Optional[str] = None) -> str:
    """
    PicRecog DB에서 자연어 질의를 바탕으로 유사한 이미지를 검색합니다.
    query: 사용자의 검색어 (예: '산 풍경', '웃고 있는 사람')
    category: 필터링할 카테고리 (people, animal, place, flower, nature, text, events)
    """
    vector_service = VectorSearchService(embedding_client=emb_client())
    results = await vector_service._search_by_text_embedding_query(
        query_text=query, category_filter=category, top_k=10  # 후보군을 늘려 에이전트가 필터링할 여지 제공
    )

    if not results:
        return "검색 결과가 없습니다."

    formatted_results = []
    for r in results:
        formatted_results.append(
            f"- ID: {r['id']}\n"
            f"- 경로: {r['path']}\n"
            f"  유사도: {r['similarity']*100:.1f}%\n"
            f"  설명: {r['description']}"
        )

    return "\n".join(formatted_results)


# 사용 가능한 모든 도구 목록
PICRECOG_TOOLS = [search_images_tool]
