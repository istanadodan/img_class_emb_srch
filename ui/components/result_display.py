"""Result Display Component - 분류 결과 표시 컴포넌트"""

import streamlit as st
from typing import List
from shared.classification_schema import ClassificationResult


def display_classification_results(results: List[ClassificationResult]):
    """분류 결과를 그룹별로 표시"""

    # 카테고리별로 결과 그룹화
    grouped = {}
    for result in results:
        category = result.category
        if category not in grouped:
            grouped[category] = []
        grouped[category].append(result)

    # 탭으로 카테고리별 표시
    tabs = st.tabs([f"{cat.value} ({len(items)}개)" for cat, items in grouped.items()])

    for tab, (category, items) in zip(tabs, grouped.items()):
        with tab:
            cols = st.columns(3)
            for idx, result in enumerate(items):
                with cols[idx % 3]:
                    st.write(f"**신뢰도**: {result.confidence:.1%}")
                    st.caption(result.image_path.split("/")[-1])
