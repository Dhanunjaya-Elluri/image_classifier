"""UI pages for the Streamlit application"""

import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
from .components import (
    create_predictions_plot,
    display_model_metrics,
)
from ..services.api import APIService
import plotly.graph_objects as go
import time

api_service = APIService()


async def classification_page():
    """Classification page content"""
    st.markdown(
        '<p class="description">🔍 Upload an image to classify it using our AI model. '
        "Powered by SqueezeNet, this tool can identify objects across 1000 categories.</p>",
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            '<div class="image-label">📸 Original Image</div>', unsafe_allow_html=True
        )
        uploaded_file = st.file_uploader(
            "Drag and drop file here",
            type=["jpg", "jpeg", "png"],
            help="Limit 200MB per file • PNG, JPG, JPEG",
        )

        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True)

            _, button_col, _ = st.columns([1, 2, 1])
            with button_col:
                classify_button = st.button("Classify Image", use_container_width=True)

    with col2:
        if uploaded_file and classify_button:
            try:
                with st.spinner("Processing image..."):
                    prediction = await api_service.predict(uploaded_file.getvalue())

                    # Create and display the predictions plot
                    fig = create_predictions_plot(prediction.predictions)
                    st.plotly_chart(fig, use_container_width=True)

                    # Display top prediction as a metric
                    top_prediction = prediction.predictions[0]
                    st.metric(
                        "Top Prediction",
                        top_prediction.class_name,
                        f"{top_prediction.confidence:.1%}",
                    )
            except ConnectionError as e:
                st.error(str(e))
                st.info("Please make sure the FastAPI server is running")
        else:
            st.markdown(
                '<div class="image-label">🎯 Classification Results</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div style="color: #C6CDD5; text-align: center; padding: 2rem;">'
                'Upload an image and click "Classify Image" to see the results here.'
                "</div>",
                unsafe_allow_html=True,
            )


async def model_info_page():
    """Model information page content"""
    try:
        info = await api_service.get_model_info()
        display_model_metrics(info)

        # Model description
        st.markdown("### Model Description")
        st.write(info.description)

        # Technical details
        st.markdown("### Technical Details")
        st.json(
            {
                "Model Type": "CNN",
                "Framework": "ONNX Runtime",
                "Input Size": f"{info.input_shape[2]}×{info.input_shape[3]} pixels",
                "Number of Classes": info.output_shape[1],
                "Input Channels": "RGB (3 channels)",
                "Preprocessing": "Normalization (0-1), NCHW format",
            }
        )
    except ConnectionError as e:
        st.error(str(e))
        st.info("Please make sure the FastAPI server is running")


async def monitoring_page():
    """Monitoring page content"""
    from ..services.monitoring import MonitoringService
    import time

    monitoring_service = MonitoringService()

    # Add refresh rate selector
    refresh_rate = st.sidebar.slider(
        "Refresh Rate (seconds)",
        min_value=1,
        max_value=60,
        value=5,
        help="How often to refresh the metrics",
        key="refresh_rate_slider",
    )

    try:
        # Create placeholder for metrics
        metrics_container = st.empty()

        while True:
            with metrics_container.container():
                metrics = await monitoring_service.get_metrics()
                current_time = int(time.time())  # Get current timestamp

                # Display request metrics
                st.subheader("Request Statistics", divider="rainbow")

                col1, col2, col3 = st.columns(3)

                total_requests = metrics["requests"]["total"]

                col1.metric(
                    "Total Predictions",
                    f"{int(total_requests):,}",
                    help="Total number of predictions made",
                )

                if total_requests > 0:
                    success_rate = metrics["requests"]["success"] / total_requests * 100
                    col2.metric(
                        "Success Rate",
                        f"{success_rate:.1f}%",
                        help="Percentage of successful predictions",
                    )

                    col3.metric(
                        "Error Rate",
                        f"{(100-success_rate):.1f}%",
                        help="Percentage of failed predictions",
                    )
                else:
                    col2.metric("Success Rate", "N/A", help="No predictions made yet")
                    col3.metric("Error Rate", "N/A", help="No predictions made yet")

                # Display response time distribution
                st.subheader("Response Time Distribution", divider="rainbow")

                if metrics["response_times"]:
                    df = pd.DataFrame(metrics["response_times"])

                    # Create range labels
                    df["range"] = pd.Series(
                        [f"0-{df['bucket'][0]}s"]
                        + [
                            f"{df['bucket'][i-1]}-{v}s"
                            for i, v in enumerate(df["bucket"][1:], 1)
                        ]
                    )

                    fig = go.Figure(
                        go.Bar(
                            x=df["range"],
                            y=df["count"],
                            text=[f"{int(count):,}" for count in df["count"]],
                            textposition="auto",
                        )
                    )

                    fig.update_layout(
                        title="Response Time Distribution",
                        xaxis_title="Response Time Range",
                        yaxis_title="Number of Requests",
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="white"),
                        showlegend=False,
                    )

                    # Use timestamp in key to make it unique for each update
                    st.plotly_chart(
                        fig,
                        use_container_width=True,
                        key=f"latency_histogram_{current_time}",
                    )
                else:
                    st.info("No prediction requests have been made yet.")

            # Wait for the specified refresh interval
            time.sleep(refresh_rate)

    except ConnectionError as e:
        st.error(str(e))
        st.info("Please make sure the FastAPI server is running")
