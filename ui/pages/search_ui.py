"""Search Page - 유사 이미지 검색"""

import streamlit as st
from backend.service.image_search import ImgSearchService
from backend.clients.embedding_client import get_studio_embedding_client as get_emb_client
from ui.components.image_utils import render_safe_image
import numpy as np
from pathlib import Path
from shared.constants import IMAGE_CATEGORIES, IMAGE_EMOJIS


async def render_search_page(path_input: str = ""):
    """유사 이미지 검색 페이지 렌더링"""
    st.title("🔍 유사 이미지 검색")
    st.write("선택한 이미지와 유사한 이미지를 카테고리별로 표시합니다.")

    # 1. 검색 경로 입력부
    with st.container(border=True):
        col1, col2 = st.columns([4, 1])
        with col1:
            current_path = st.text_input(
                "이미지 파일 경로",
                value=path_input,
                placeholder="예: C:\\Users\\Pictures\\photo.jpg",
                label_visibility="collapsed",
            )
        with col2:
            search_clicked = st.button("검색", type="primary", width="stretch")
            if path_input and st.button("초기화", width="stretch"):
                st.rerun()

    if not current_path and not search_clicked:
        st.info("💡 검색할 이미지의 전체 경로를 입력하고 [검색] 버튼을 눌러주세요.")
        return

    # 2. 검색 수행 및 탭 렌더링
    service = ImgSearchService(embedding_client=get_emb_client())

    # 카테고리 매핑 (동적 생성)
    tabs = st.tabs(
        [f"{IMAGE_EMOJIS.get(cid, '')} {name}" for cid, name in IMAGE_CATEGORIES.items()]
    )

    for i, (cat_id, cat_name) in enumerate(IMAGE_CATEGORIES.items()):
        display_name = f"{IMAGE_EMOJIS.get(cat_id, '')} {cat_name}"
        with tabs[i]:
            with st.spinner(f"'{display_name}' 유사 이미지 검색 중..."):
                results = service.search_by_path(image_path=current_path, category_filter=cat_id)

            if not results:
                st.info(f"'{display_name}' 카테고리에서 유사한 이미지를 찾을 수 없습니다.")
                continue

            st.write(f"유사한 **{cat_name}** 이미지 ({len(results)}건)")

            # 3열 갤러리 렌더링
            cols = st.columns(3)
            for idx, img in enumerate(results):
                col_idx = idx % 3
                with cols[col_idx]:
                    render_safe_image(
                        img["path"],
                        width="stretch",
                    )
                    st.caption(f"📍 {Path(img['path']).name}")
                    st.metric("유사도", f"{img['similarity']*100:.1f}%")
                    with st.expander("상세 정보"):
                        st.json(
                            {
                                "category": img.get("category"),
                                "confidence": img.get("confidence"),
                                "description": img.get("description"),
                            }
                        )

    # 3. 쿼리 이미지 정보 (하단)
    if current_path:
        st.divider()
        st.caption(f"🔍 쿼리 이미지: {current_path}")
