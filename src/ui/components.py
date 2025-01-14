"""UI components for the Streamlit application"""

from typing import List

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ..api.schemas import ModelInfo, PredictionItem


def create_predictions_plot(predictions: List[PredictionItem]) -> go.Figure:
    """Create a horizontal bar plot for predictions"""
    df = pd.DataFrame(
        [{"class_name": p.class_name, "confidence": p.confidence} for p in predictions]
    )

    fig = go.Figure(
        go.Bar(
            x=df["confidence"],
            y=df["class_name"],
            orientation="h",
            text=[f"{conf:.1%}" for conf in df["confidence"]],
            textposition="auto",
            marker=dict(
                color="#0078D4", line=dict(color="rgba(255, 255, 255, 0.5)", width=1)
            ),
        )
    )

    fig.update_layout(
        title=dict(
            text="Top 10 Predictions",
            x=0.5,
            y=0.95,
            xanchor="center",
            yanchor="top",
            font=dict(size=20, color="white"),
        ),
        xaxis_title="Confidence",
        yaxis_title="Class",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        xaxis=dict(
            gridcolor="rgba(255,255,255,0.1)",
            tickformat=".1%",
            showgrid=True,
            range=[0, 1],
            dtick=0.1,
            tickmode="linear",
            ticktext=[f"{i*10}%" for i in range(11)],
            tickvals=[i / 10 for i in range(11)],
        ),
        yaxis=dict(gridcolor="rgba(255,255,255,0.1)", autorange="reversed"),
        margin=dict(l=20, r=20, t=60, b=20),
        height=400,
    )

    return fig


def display_model_metrics(info: ModelInfo) -> None:
    """Display model metrics in columns"""
    col1, col2, col3 = st.columns(3)
    col1.metric("Model Architecture", info.name)
    col2.metric("Input Shape", " × ".join(map(str, info.input_shape)))
    col3.metric("Output Shape", " × ".join(map(str, info.output_shape)))
