"""Theme Configuration - Streamlit 테마 설정"""

import streamlit as st


def apply_theme():
    """따뜻하고 현대적인 테마 적용"""

    st.set_page_config(
        page_title="PicRecog - 이미지 분류",
        page_icon="🖼️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.navigation(
        [
            st.Page("app.py", title="🏠 분류"),
            # st.Page("pages/gallery_ui.py", title="🖼️ 갤러리"),
            # st.Page("pages/search_ui.py", title="🔍 검색"),
            # st.Page("pages/classifier_ui.py", title="⚙️ 분류"),
        ]
    )

    # Custom CSS for warm and modern feel
    st.markdown(
        """
        <style>
        :root {
            --primary-color: #FF6B6B;
            --secondary-color: #4ECDC4;
            --accent-color: #FFE66D;
            --bg-color: #F7F9FB;
            --text-color: #2C3E50;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-color);
        }

        .main {
            background: linear-gradient(135deg, #F7F9FB 0%, #EFF3F8 100%);
        }

        h1, h2, h3 {
            color: var(--text-color);
            font-weight: 600;
        }

        .stButton > button {
            border-radius: 8px;
            border: none;
            padding: 10px 24px;
            background: linear-gradient(135deg, #FF6B6B 0%, #FF8E8E 100%);
            color: white;
            font-weight: 600;
            transition: all 0.3s ease;
        }

        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(255, 107, 107, 0.3);
        }

        .stContainer {
            border-radius: 12px;
            padding: 20px;
            background: white;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        }

        /* Skeleton Pulse Animation */
        @keyframes pulse {
            0% { opacity: 0.6; }
            50% { opacity: 1; }
            100% { opacity: 0.6; }
        }
        .skeleton {
            animation: pulse 1.5s infinite ease-in-out;
            background-color: #F1F3F5;
            border-radius: 8px;
            width: 100%;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
