"""Main Streamlit application"""

import streamlit as st
from src.ui.pages import classification_page, model_info_page, monitoring_page
from src.core.config import settings
from src.ui.styles import CUSTOM_CSS

# Configure page settings
st.set_page_config(
    layout="wide",
    page_title=settings.PROJECT_NAME,
    initial_sidebar_state="collapsed",
)

# Apply custom CSS
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


async def main():
    """Main application entry point"""
    st.markdown(
        '<h1 class="main-header">Image Classification</h1>', unsafe_allow_html=True
    )

    # Create tabs
    tab1, tab2, tab3 = st.tabs(
        ["🔍 Classification", "ℹ️ Model Information", "📊 Monitoring"]
    )

    # Content for each tab
    with tab1:
        await classification_page()
    with tab2:
        await model_info_page()
    with tab3:
        await monitoring_page()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
