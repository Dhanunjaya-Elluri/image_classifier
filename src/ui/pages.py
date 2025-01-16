"""UI pages for the Streamlit application"""

import time

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from src.core.exceptions import (
    APIConnectionError,
    ModelError,
    PrometheusConnectionError,
    ValidationError,
)
from src.services.api import APIService
from src.services.monitoring import MonitoringService
from src.ui.components import create_predictions_plot

api_service = APIService()


async def classification_page() -> None:
    """Classification page"""
    _, center_col, _ = st.columns([1, 2, 1])

    with center_col:
        try:
            # Test API connection
            await api_service.get_model_info()

            st.markdown(
                "<p>Upload an image to classify</p>",
                unsafe_allow_html=True,
            )

            uploaded_file = st.file_uploader(
                "Upload Image",
                type=["jpg", "jpeg", "png"],
                help="Supported formats: JPG, JPEG, PNG",
                label_visibility="hidden",
            )

            result_container = st.empty()

            if uploaded_file is not None:
                try:
                    with st.spinner("Analyzing image..."):
                        image = Image.open(uploaded_file)
                        uploaded_file.seek(0)
                        response = await api_service.predict(uploaded_file.read())

                        if response and response.predictions:
                            with result_container.container():
                                top_prediction = response.predictions[0]
                                formatted_class = top_prediction.class_name.replace(
                                    "_", " "
                                ).title()

                                st.markdown(
                                    "<h2 style='text-align: center;'>Classification Result</h2>",
                                    unsafe_allow_html=True,
                                )
                                st.markdown(
                                    f"#### Predicted Class: <span style='color: #FF5733;'>{formatted_class}</span>, "
                                    f"Confidence Score: <span style='color: #FF5733;'>{top_prediction.confidence:.1%}</span>",
                                    unsafe_allow_html=True,
                                )

                                # Show the image
                                st.image(
                                    image,
                                    use_container_width=True,
                                    caption="Uploaded Image",
                                )

                                # Show top 10 predictions
                                with st.expander("Show Top 10 Predictions"):
                                    predictions_fig = create_predictions_plot(
                                        response.predictions
                                    )
                                    st.plotly_chart(
                                        predictions_fig,
                                        use_container_width=True,
                                        config={"displayModeBar": False},
                                    )

                except (APIConnectionError, ModelError, ValidationError) as e:
                    st.error(f"Error: {str(e)}")
                except Exception as e:
                    st.error(f"An unexpected error occurred: {str(e)}")

        except APIConnectionError:
            st.error("Unable to connect to the API server")
            st.warning(
                "Please make sure the FastAPI server is running. You can start it with:\n"
                "```bash\n"
                "uvicorn src.api.main:app --reload --port 8000\n"
                "```"
            )


async def model_info_page() -> None:
    """Model information page content"""
    try:
        info = await api_service.get_model_info()

        st.markdown("### Model Description")
        st.write(info.description)

        st.markdown("### Technical Details")

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
        st.error("Unable to connect to the API server")
        st.warning(
            "Please make sure the FastAPI server is running. You can start it with:\n"
            "```bash\n"
            "uvicorn src.api.main:app --reload --port 8000\n"
            "```"
        )


async def monitoring_page() -> None:
    """Monitoring page content"""
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
        metrics_container = st.empty()

        while True:
            with metrics_container.container():
                try:
                    metrics = await monitoring_service.get_metrics()
                    current_time = int(time.time())

                    # Create two columns for the layout
                    left_col, right_col = st.columns(2)

                    # Left Column: Request Statistics
                    with left_col:
                        st.subheader("Request Statistics", divider="blue")
                        total_requests = metrics["requests"]["total"]

                        st.metric(
                            "Total Predictions",
                            f"{int(total_requests):,}",
                            help="Total number of predictions made",
                        )

                        if total_requests > 0:
                            success_rate = (
                                metrics["requests"]["success"] / total_requests * 100
                            )
                            st.metric(
                                "Success Rate",
                                f"{success_rate:.1f}%",
                                help="Percentage of successful predictions",
                            )
                            st.metric(
                                "Error Rate",
                                f"{(100-success_rate):.1f}%",
                                help="Percentage of failed predictions",
                            )
                        else:
                            st.metric(
                                "Success Rate",
                                "N/A",
                                help="No predictions made yet",
                            )
                            st.metric(
                                "Error Rate",
                                "N/A",
                                help="No predictions made yet",
                            )

                    # Right Column: Response Time Distribution
                    with right_col:
                        st.subheader("Response Time Distribution", divider="blue")
                        if metrics["response_times"]:
                            df = pd.DataFrame(metrics["response_times"])

                            # Create range labels
                            ranges = []
                            for i, row in enumerate(df.itertuples()):
                                if i == 0:
                                    ranges.append("0-100ms")
                                else:
                                    prev_bucket = df["bucket"].iloc[i - 1]
                                    curr_bucket = row.bucket
                                    if curr_bucket == float("inf"):
                                        ranges.append(">1000ms")
                                    else:
                                        ranges.append(
                                            f"{int(prev_bucket*1000)}-{int(curr_bucket*1000)}ms"
                                        )

                            df["range"] = ranges

                            fig = go.Figure(
                                go.Bar(
                                    x=df["range"],
                                    y=df["count"],
                                    text=[f"{int(count):,}" for count in df["count"]],
                                    textposition="auto",
                                    marker=dict(
                                        color="#0078D4",
                                        line=dict(
                                            color="rgba(255, 255, 255, 0.5)", width=1
                                        ),
                                    ),
                                )
                            )

                            fig.update_layout(
                                xaxis_title="Response Time (milliseconds)",
                                yaxis_title="Number of Requests",
                                plot_bgcolor="rgba(0,0,0,0)",
                                paper_bgcolor="rgba(0,0,0,0)",
                                font=dict(color="white"),
                                showlegend=False,
                                margin=dict(l=20, r=20, t=20, b=20),
                                height=350,
                            )

                            st.plotly_chart(
                                fig,
                                use_container_width=True,
                                key=f"latency_histogram_{current_time}",
                            )
                        else:
                            st.info("No prediction requests have been made yet.")

                except PrometheusConnectionError:
                    st.error("Unable to connect to Prometheus server")
                    st.warning(
                        "Please make sure Prometheus is running. You can start it with:\n"
                        "```bash\n"
                        "prometheus --config.file=prometheus.local.yml\n"
                        "```"
                    )

            time.sleep(refresh_rate)

    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        st.info("Please try refreshing the page")
