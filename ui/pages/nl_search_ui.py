"""Natural Language Search Page - 자연어 기반 이미지 검색"""

import streamlit as st
import asyncio
from backend.service.image_search import ImgSearchService
from backend.clients.embedding_client import get_studio_embedding_client as get_emb_client
from ui.components.image_utils import render_safe_image
from pathlib import Path


async def render_nl_search_page():
    """자연어 검색 페이지 렌더링"""
    st.title("🤖 AI 자연어 이미지 검색")
    st.write("이미지의 내용을 설명하여 검색해 보세요. (예: '비 오는 날의 도시', '빨간 자동차')")

    # 1. 자연어 검색어 입력
    with st.container(border=True):
        col1, col2 = st.columns([4, 1])
        with col1:
            query_text = st.text_input(
                "무엇을 찾고 싶으신가요?",
                placeholder="예: 웃고 있는 사람들의 사진",
                label_visibility="collapsed",
            )
        with col2:
            search_clicked = st.button("AI 검색", type="primary", width="stretch")

    if not query_text and not search_clicked:
        return

    # 2. 검색 수행
    service = ImgSearchService(embedding_client=get_emb_client())

    with st.spinner("AI가 이미지를 분석하여 검색 중입니다..."):
        try:
            results = await service.search_by_text(query_text=query_text)

            if not results:
                st.warning(f"'{query_text}'와 유사한 이미지를 찾을 수 없습니다.")
                return

            st.success(f"총 {len(results)}개의 이미지를 찾았습니다.")

            # 3열 갤러리 렌더링
            cols = st.columns(3)
            for idx, img in enumerate(results):
                col_idx = idx % 3
                with cols[col_idx]:
                    render_safe_image(img["path"], width="stretch")
                    st.caption(f"📍 {Path(img['path']).name}")
                    st.progress(img["similarity"], text=f"유사도: {img['similarity']*100:.1f}%")

                    with st.expander("AI 분석 내용"):
                        if img.get("justification"):
                            st.info(f"**판단 근거:** {img['justification']}")
                        st.write(f"**카테고리:** {img['category']}")
                        st.write(f"**설명:** {img['description']}")
                        if img.get("objects"):
                            st.write(f"**검출 객체:** {img['objects']}")

        except Exception as e:
            st.error(f"검색 중 오류가 발생했습니다: {str(e)}")

    st.divider()
    st.caption("※ 자연어 검색은 이미지의 시맨틱 정보를 바탕으로 수행됩니다.")
