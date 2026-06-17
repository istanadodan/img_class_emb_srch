"""Gallery Page - 갤러리 스타일 이미지 표시"""

import streamlit as st
from backend.service.image_search import ImgSearchService
from backend.clients.embedding_client import get_studio_embedding_client as get_emb_client
from ui.components.image_utils import render_safe_image


from shared.constants import IMAGE_CATEGORIES, IMAGE_EMOJIS


async def render_gallery_page():
    """갤러리 페이지 렌더링"""
    st.title("🖼️ 갤러리")

    # 세션 상태 초기화 (유사 검색 대상 ID 저장용)
    if "active_search_id" not in st.session_state:
        st.session_state.active_search_id = None

    st.write("분류된 이미지를 갤러리로 표시합니다.")

    # 카테고리 필터 (동적 생성)
    st.write("### 🔍 필터")
    selected_categories = []

    # 7개 카테고리를 한 줄에 표시하기 위해 컬럼 생성
    cols = st.columns(len(IMAGE_CATEGORIES))
    for i, (cat_id, cat_name) in enumerate(IMAGE_CATEGORIES.items()):
        with cols[i]:
            emoji = IMAGE_EMOJIS.get(cat_id, "📁")
            if st.checkbox(f"{emoji} {cat_name}", value=True, key=f"cat_{cat_id}"):
                selected_categories.append(cat_id)

    # 서비스 초기화
    service = ImgSearchService(embedding_client=get_emb_client())

    # 데이터베이스에서 이미지 목록 조회
    images = service.get_all_images(categories=selected_categories if selected_categories else None)

    if not images:
        st.info("표시할 이미지가 없습니다. 먼저 이미지를 분류해주세요.")
        return

    st.write(f"총 **{len(images)}**개의 이미지가 표시됩니다.")
    st.write("---")

    # 그리드 레이아웃 (9열)
    cols_per_row = 9

    # 이미지를 행 단위로 루프
    for i in range(0, len(images), cols_per_row):
        row_images = images[i : i + cols_per_row]
        cols = st.columns(cols_per_row)

        selected_img_in_row = None

        # 1. 현재 행의 이미지들 렌더링
        for idx, img in enumerate(row_images):
            with cols[idx]:
                render_safe_image(
                    img["path"],
                    caption=f"{img['category']}",
                    width="stretch",
                )
                st.caption(f"{img['category']}:\n{img['path'].split('\\')[-1].split(".")[0]}")

                # 유사 검색 버튼
                if st.button("검색", key=f"gallery_btn_{img['id']}"):
                    st.session_state.active_search_id = img["id"]
                    st.rerun()  # 클릭 시 리런하여 결과창 표시

            # 현재 행에 선택된 이미지가 있는지 확인
            if st.session_state.active_search_id == img["id"]:
                selected_img_in_row = img

        # 2. 만약 이 행에 선택된 이미지가 있다면, 행 바로 하단에 결과 출력
        if selected_img_in_row:
            with st.container():
                st.write("---")
                st.markdown(f"#### 🔍 `{selected_img_in_row['category']}` 관련 유사 이미지")

                # 결과 닫기 버튼
                if st.button("✖️ 결과 닫기", key=f"close_res_{selected_img_in_row['id']}"):
                    st.session_state.active_search_id = None
                    st.rerun()

                with st.spinner("유사한 이미지를 검색 중..."):
                    # 유사 이미지 검색 수행 (선택된 이미지의 카테고리를 필터로 전달)
                    results = service.search_by_path(
                        image_path=selected_img_in_row["path"],
                        category_filter=selected_img_in_row["category"],
                    )

                if not results:
                    st.warning("유사한 이미지를 찾을 수 없습니다.")
                else:
                    # 검색 결과를 5열로 작게 표시
                    res_cols = st.columns(5)
                    for r_idx, r_img in enumerate(results):
                        with res_cols[r_idx % 5]:
                            render_safe_image(
                                r_img["path"],
                                caption=f"유사도: {r_img['similarity']*100:.1f}%",
                                width="stretch",
                            )
                st.write("---")
