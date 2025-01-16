"""UI components"""

from typing import List

import pandas as pd
import plotly.graph_objects as go

from src.api.schemas import PredictionItem


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
                color="#0078D4",
                line=dict(color="rgba(255, 255, 255, 0.5)", width=1),
            ),
        )
    )

    fig.update_layout(
        xaxis_title="Confidence",
        yaxis_title="Class",
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
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        autosize=True,
        margin=dict(l=20, r=20, t=60, b=20),
        height=400,
    )

    return fig
