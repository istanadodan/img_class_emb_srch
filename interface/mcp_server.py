"""MCP Server for PicRecog - Claude Desktop 연동 (stdio) 및 자연어 검색 도구 제공"""

import asyncio
import logging
from fastmcp import FastMCP
from backend.service.image_search import ImgSearchService
from backend.clients.embedding_client import get_studio_embedding_client as get_emb_client
from shared.constants import IMAGE_CATEGORIES
from typing import Optional

# 로그 설정 (표준 에러로 출력하여 stdio 통신 방해 금지)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mcp-picrecog")

# FastMCP 서버 인스턴스 생성 (Claude Desktop은 stdio를 통해 통신함)
mcp = FastMCP("PicRecog")


@mcp.tool()
async def search_images(query: str, category: Optional[str] = None) -> str:
    """
    PicRecog DB에서 사용자의 자연어 질의를 바탕으로 유사한 이미지를 검색합니다.

    Args:
        query: 검색어 (예: "비오는 날 찍은 사진", "강아지가 있는 풍경")
        category: 필터링할 카테고리 (people, nature, text, events)
    """
    try:
        service = ImgSearchService(embedding_client=get_emb_client())
        # DB 검색 수행 (await 필요)
        results = await service.search_by_text(query_text=query, category_filter=category)

        if not results:
            return f"'{query}'에 대한 검색 결과가 없습니다."

        # 결과를 문자열로 포맷팅하여 반환
        output = [f"총 {len(results)}건의 유사 이미지를 찾았습니다:\n"]
        for res in results:
            output.append(
                f"- 경로: {res['path']}\n"
                f"  카테고리: {res['category']} (유사도: {res['similarity']*100:.1f}%)\n"
                f"  설명: {res['description']}\n"
            )

        return "\n".join(output)

    except Exception as e:
        logger.error(f"MCP Tool Error: {str(e)}")
        return f"검색 중 오류가 발생했습니다: {str(e)}"


if __name__ == "__main__":
    # Claude Desktop 연동을 위해 기본 stdio 방식으로 실행
    mcp.run(transport="stdio")
