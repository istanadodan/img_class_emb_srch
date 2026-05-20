"""Classifier Page - 이미지 분류 메인 페이지"""

import streamlit as st
from pathlib import Path


def render_classifier_page():
    """이미지 분류 페이지 렌더링"""

    st.title("🖼️ 이미지 분류")
    st.write("이미지 경로 또는 폴더를 입력하여 자동으로 분류합니다.")

    with st.container(border=True):
        col1, col2 = st.columns(2)

        with col1:
            input_type = st.radio(
                "입력 방식 선택",
                ["📁 폴더", "🖼️ 이미지 파일"],
                label_visibility="collapsed",
            )

        with col2:
            if input_type == "📁 폴더":
                path_input = st.text_input(
                    "폴더 경로를 입력하세요",
                    placeholder="예: C:\\Users\\Pictures\\MyPhotos",
                )
            else:
                path_input = st.text_input(
                    "이미지 파일 경로를 입력하세요",
                    placeholder="예: C:\\Users\\Pictures\\photo.jpg",
                )

    if st.button("분류 시작", use_container_width=True, type="primary"):
        if not path_input:
            st.error("경로를 입력해주세요.")
        else:
            with st.spinner("분류 중..."):
                # TODO: 실제 분류 로직 연결
                st.success("분류 완료!")


if __name__ == "__main__":
    render_classifier_page()
