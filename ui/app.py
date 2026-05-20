"""Main Streamlit Application - PicRecog UI 진입점"""

import streamlit as st
from ui.styles.theme import apply_theme
from ui.pages.classifier import render_classifier_page


def main():
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
            ["🏠 분류", "⚙️ 설정", "📊 통계"],
            label_visibility="collapsed",
        )

        st.divider()
        st.caption("PicRecog v0.1.0")

    # 페이지 라우팅
    if page == "🏠 분류":
        render_classifier_page()
    elif page == "⚙️ 설정":
        st.title("⚙️ 설정")
        st.write("설정 페이지 개발 중...")
    elif page == "📊 통계":
        st.title("📊 통계")
        st.write("통계 페이지 개발 중...")


if __name__ == "__main__":
    main()
