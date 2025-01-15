"""Main Streamlit application"""

import streamlit as st

from src.core.config import settings
from src.ui.pages import classification_page, model_info_page, monitoring_page
from src.ui.styles import CUSTOM_CSS

# Configure page settings
st.set_page_config(
    layout="wide",
    page_title=settings.PROJECT_NAME,
)

# Apply custom CSS
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Add CSS for centered tabs and description
st.markdown(
    """
    <style>
        .stTabs [data-baseweb="tab-list"] {
            gap: 24px;
            justify-content: center;
        }
        .project-description {
            color: #718096;
            text-align: center;
            max-width: 800px;
            margin: 0 auto 32px auto;
            line-height: 1.6;
            font-size: 1.1em;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


async def main():
    """Main application entry point"""
    st.markdown(
        '<h1 class="main-header" style="text-align: center; margin-bottom: 16px;">Image Classification</h1>',
        unsafe_allow_html=True,
    )

    # Add project description
    st.markdown(
        """
        <div class="project-description">
            This is a simple image classification tool that uses a SqueezeNet1.1 model that can classify images into 1000 different
            categories which you find the list <a href="https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt" target="_blank">here</a>.
            For more information, please visit the <a href="https://github.com/onnx/models/tree/main/validated/vision/classification/squeezenet" target="_blank">SqueezeNet model documentation</a>.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Classification", "Model Information", "Monitoring"])

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
