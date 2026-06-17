"""Main Streamlit Application - PicRecog UI 진입점"""

import streamlit as st
import asyncio
from ui.styles.theme import apply_theme
from ui.pages.classifier_ui import render_classifier_page
from ui.pages.gallery_ui import render_gallery_page
from ui.pages.search_ui import render_search_page
from ui.pages.nl_search_ui import render_nl_search_page

# from backend.clients.embedding_client import preload_emb_client
from backend.system import database
from backend.domain import db_models  # DB 모델 등록을 위해 명시적 임포트
from backend.system.config import settings
import logging
import warnings

# 로깅 설정 초기화
settings.setup_logging()
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=FutureWarning)


async def main():
    """메인 애플리케이션"""

    # 테마 적용
    apply_theme()

    # 사이드바
    with st.sidebar:
        st.title("🎯 PicRecog")
        st.write("AI 기반 이미지 자동 분류 시스템")

        st.divider()

        page = st.radio(
            "메뉴",
            ["🏠 분류", "🖼️ 갤러리", "🔍 검색", "🤖 AI 검색", "⚙️ 설정", "📊 통계"],
            label_visibility="collapsed",
        )

        st.divider()
        st.caption("PicRecog v0.1.0")

    # 페이지 라우팅
    if page == "🏠 분류":
        await render_classifier_page()
    elif page == "🖼️ 갤러리":
        await render_gallery_page()
    elif page == "🔍 검색":
        await render_search_page()
    elif page == "🤖 AI 검색":
        await render_nl_search_page()
    elif page == "⚙️ 설정":
        st.title("⚙️ 설정")
        st.write("설정 페이지 개발 중...")
    elif page == "📊 통계":
        st.title("📊 통계")
        st.write("통계 페이지 개발 중...")


if __name__ == "__main__":
    # 테이블 생성
    database.init_db()
    # preload_emb_client()
    asyncio.run(main())
