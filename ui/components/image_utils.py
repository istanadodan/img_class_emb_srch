import streamlit as st
import os
import base64
from pathlib import Path

# "No Image" placeholder SVG - 브랜드 톤에 맞춘 경량화된 이미지
NO_IMAGE_SVG = """
<svg width="200" height="200" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
    <rect width="24" height="24" fill="#F8F9FA"/>
    <path d="M21 19V5C21 3.9 20.1 3 19 3H5C3.9 3 3 3.9 3 5V19C3 20.1 3.9 21 5 21H19C20.1 21 21 20.1 21 19ZM8.5 13.5L11 16.51L14.5 12L19 18H5L8.5 13.5Z" fill="#CED4DA"/>
    <text x="12" y="21" font-family="sans-serif" font-size="2" fill="#ADB5BD" text-anchor="middle">FILE NOT FOUND</text>
</svg>
"""


def get_placeholder_image() -> str:
    """Placeholder 이미지를 base64 데이터 URI로 반환"""
    b64 = base64.b64encode(NO_IMAGE_SVG.encode()).decode()
    return f"data:image/svg+xml;base64,{b64}"


def render_safe_image(
    image_path: str, caption: str | None = None, use_container_width: bool = True, **kwargs
):
    """
    이미지 존재 여부를 확인하고 안전하게 렌더링 (정책 5번 적용)

    1. Skeleton: 로딩 중 또는 확인 중 상태를 빈 컨테이너로 확보
    2. Validation: 실제 파일 경로 존재 여부 확인
    3. Missing State: 파일 부재 시 'No Image' 플레이스홀더 표시
    """
    # 1. 컨테이너 확보 (Skeleton UI 효과를 위한 st.empty 사용)
    placeholder = st.empty()

    # 2. 로딩/전환 효과 시뮬레이션 (CSS 기반 Skeleton)
    # Streamlit의 렌더링 특성상 매우 짧게 노출되지만, 부재 확인 로직과의 구분감을 줌
    with placeholder.container():
        st.markdown(
            '<div class="skeleton" style="height: 150px; margin-bottom: 10px;"></div>',
            unsafe_allow_html=True,
        )

    # 3. 파일 존재 여부 확인
    is_valid = False
    if image_path:
        path_obj = Path(image_path)
        if path_obj.exists() and path_obj.is_file():
            is_valid = True

    # 4. 최종 결과 렌더링
    if is_valid:
        placeholder.image(
            image_path, caption=caption, use_container_width=use_container_width, **kwargs
        )
    else:
        # 부재가 확정된 경우에만 대체 이미지 노출 (정책 5번 핵심)
        placeholder.image(
            get_placeholder_image(),
            caption=f"⚠️ {caption if caption else '이미지 유실'}",
            use_container_width=use_container_width,
            **kwargs,
        )
