"""UI pages for the Streamlit application"""

import time

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from ..core.exceptions import (
    APIConnectionError,
    ModelError,
    PrometheusConnectionError,
    ValidationError,
)
from ..services.api import APIService
from .components import create_predictions_plot

api_service = APIService()


async def classification_page():
    """Classification page content"""

    # Initialize session state for results
    if "results" not in st.session_state:
        st.session_state.results = None
    if "previous_file_state" not in st.session_state:
        st.session_state.previous_file_state = None

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            '<div class="image-label" style="text-align: center;">Original Image</div>',
            unsafe_allow_html=True,
        )

        uploaded_file = st.file_uploader(
            "Upload the image file here",
            type=["jpg", "jpeg", "png"],
            help="Limit 200MB per file • PNG, JPG, JPEG",
            key="image_uploader",
        )

        # Check if file was removed
        if st.session_state.previous_file_state is not None and uploaded_file is None:
            st.session_state.previous_file_state = None
            st.rerun()

        # Update previous state
        st.session_state.previous_file_state = uploaded_file

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True)

            _, button_col, _ = st.columns([1, 2, 1])
            with button_col:
                classify_button = st.button("Classify Image", use_container_width=True)

    with col2:
        st.markdown(
            '<div class="image-label" style="text-align: center;">Classification Results</div>',
            unsafe_allow_html=True,
        )

        results_container = st.empty()

        if uploaded_file is None:
            with results_container.container():
                st.markdown(
                    '<div style="color: #C6CDD5; text-align: center; padding: 2rem;">'
                    'Upload an image and click "Classify Image" to see the results here.'
                    "</div>",
                    unsafe_allow_html=True,
                )
        elif classify_button:
            try:
                with st.spinner("Processing image..."):
                    prediction = await api_service.predict(uploaded_file.getvalue())
                    st.session_state.results = prediction

                    with results_container.container():
                        # Top prediction with custom styling
                        top_prediction = prediction.predictions[0]
                        st.markdown(
                            f"""
                            <div style="font-size: 1.2em; color: #f8f9f9; margin-bottom: 12px;">
                                Predicted Class: <span style="color: #4ade80; font-weight: 500;">{top_prediction.class_name.replace('_', ' ').title()}</span>
                            </div>
                            <div style="font-size: 1.2em; color: #f8f9f9; margin-bottom: 12px;">
                                Confidence: <span style="color: #4ade80; font-weight: 500;">{top_prediction.confidence:.1%}</span>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        fig = create_predictions_plot(prediction.predictions)
                        st.plotly_chart(fig, use_container_width=True)

            except APIConnectionError:
                st.error("Unable to connect to the API server")
                st.warning(
                    "Please make sure the FastAPI server is running. You can start it with:\n"
                    "```bash\n"
                    "uvicorn src.api.main:app --reload --port 8000\n"
                    "```"
                )
                st.info(
                    "Open a new terminal window and run the command above to start the server"
                )
            except ModelError as e:
                st.error(f"Model Error: {str(e)}")
                st.info(
                    "Please try with a different image or check if the model is loaded"
                )
            except ValidationError as e:
                st.error(f"Invalid Input: {str(e)}")
                st.info("Please make sure you're uploading a valid image file")


async def model_info_page():
    """Model information page content"""
    try:
        info = await api_service.get_model_info()

        # Model description
        st.markdown("### Model Description")
        st.write(info.description)

        # Technical details
        st.markdown("### Technical Details")

        # Custom CSS for cards
        st.markdown(
            """
            <style>
            .tech-card {
                background-color: rgba(255, 255, 255, 0.05);
                padding: 20px;
                border-radius: 8px;
                margin: 10px 0;
                border: 1px solid rgba(255, 255, 255, 0.1);
                box-shadow: 0 1px 2px rgba(0,0,0,0.1);
                backdrop-filter: blur(4px);
            }
            .tech-label {
                color: #9ca3af;
                font-size: 0.9em;
                font-weight: 500;
                margin-bottom: 8px;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }
            .tech-value {
                color: #f8f9f9;
                font-size: 1.1em;
                font-weight: 400;
                letter-spacing: 0.2px;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        # Create three columns for technical details
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(
                """
                <div class="tech-card">
                    <div class="tech-label">Model Architecture</div>
                    <div class="tech-value">SqueezeNet 1.1 (CNN)</div>
                </div>
                <div class="tech-card">
                    <div class="tech-label">Framework</div>
                    <div class="tech-value">ONNX Runtime</div>
                </div>
            """,
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown(
                f"""
                <div class="tech-card">
                    <div class="tech-label">Input Resolution</div>
                    <div class="tech-value">{info.input_shape[2]}×{info.input_shape[3]} pixels</div>
                </div>
                <div class="tech-card">
                    <div class="tech-label">Number of Classes</div>
                    <div class="tech-value">{info.output_shape[1]:,}</div>
                </div>
            """,
                unsafe_allow_html=True,
            )

        with col3:
            st.markdown(
                """
                <div class="tech-card">
                    <div class="tech-label">Input Format</div>
                    <div class="tech-value">RGB (3 channels), NCHW</div>
                </div>
                <div class="tech-card">
                    <div class="tech-label">Preprocessing</div>
                    <div class="tech-value">Normalization (0-1), Center Crop</div>
                </div>
            """,
                unsafe_allow_html=True,
            )

    except APIConnectionError:
        st.error("🚫 Unable to connect to the API server")
        st.warning(
            "Please make sure the FastAPI server is running. You can start it with:\n"
            "```bash\n"
            "uvicorn src.api.main:app --reload --port 8000\n"
            "```"
        )
        st.info(
            "💡 Tip: Open a new terminal window and run the command above to start the server"
        )


async def monitoring_page():
    """Monitoring page content"""
    from ..services.monitoring import MonitoringService

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
                try:
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
                        success_rate = (
                            metrics["requests"]["success"] / total_requests * 100
                        )
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
                        col2.metric(
                            "Success Rate", "N/A", help="No predictions made yet"
                        )
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

                except PrometheusConnectionError:
                    st.error("🚫 Unable to connect to Prometheus server")
                    st.warning(
                        "Please make sure Prometheus is running. You can start it with:\n"
                        "```bash\n"
                        "docker-compose up prometheus\n"
                        "```"
                    )
                    st.info(
                        "💡 Tip: Make sure Docker is running and try the command above in a new terminal window"
                    )
                    break  # Exit the loop if Prometheus is not available

            # Wait for the specified refresh interval
            time.sleep(refresh_rate)

    except Exception as e:
        st.error(f"⚠️ An unexpected error occurred: {str(e)}")
        st.info(
            "Please try refreshing the page or contact support if the issue persists"
        )
