#!/usr/bin/env python3
"""
Generate poster-quality confusion matrix chart for water quality classifier.

Creates a high-resolution heatmap showing model performance suitable for
research posters.
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import plotly.graph_objects as go


def main():
    """Main execution function."""
    # Load model metadata
    metadata_path = Path("data/models/metadata_20251203_000838.json")

    print(f"Loading model metadata from {metadata_path}...")
    with open(metadata_path) as f:
        metadata = json.load(f)

    # Extract confusion matrix values
    metrics = metadata["classifier"]["metrics"]
    tp = metrics["true_positives"]   # 321 - correctly predicted SAFE
    tn = metrics["true_negatives"]   # 246 - correctly predicted UNSAFE
    fp = metrics["false_positives"]  # 6 - predicted SAFE, actually UNSAFE
    fn = metrics["false_negatives"]  # 4 - predicted UNSAFE, actually SAFE

    total = tp + tn + fp + fn
    accuracy = metrics["accuracy"]
    precision = metrics["precision"]
    recall = metrics["recall"]

    print(f"  True Positives (SAFE correct): {tp}")
    print(f"  True Negatives (UNSAFE correct): {tn}")
    print(f"  False Positives (predicted SAFE, was UNSAFE): {fp}")
    print(f"  False Negatives (predicted UNSAFE, was SAFE): {fn}")
    print(f"  Overall Accuracy: {accuracy:.2%}")

    # Create confusion matrix array
    # Rows: Actual (UNSAFE=0, SAFE=1)
    # Cols: Predicted (UNSAFE=0, SAFE=1)
    cm = np.array([
        [tn, fp],   # Actual UNSAFE: [correct UNSAFE, wrong SAFE]
        [fn, tp]    # Actual SAFE: [wrong UNSAFE, correct SAFE]
    ])

    # Calculate percentages
    cm_pct = cm / total * 100

    # Create annotation text with count and percentage
    annotations = [
        [f"{tn}<br>({cm_pct[0,0]:.1f}%)", f"{fp}<br>({cm_pct[0,1]:.1f}%)"],
        [f"{fn}<br>({cm_pct[1,0]:.1f}%)", f"{tp}<br>({cm_pct[1,1]:.1f}%)"]
    ]

    # Create figure
    print("\nCreating poster-quality confusion matrix...")

    # Custom colorscale: light to dark blue
    colorscale = [
        [0.0, "#f7fbff"],
        [0.25, "#c6dbef"],
        [0.5, "#6baed6"],
        [0.75, "#2171b5"],
        [1.0, "#084594"]
    ]

    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=["UNSAFE", "SAFE"],
        y=["UNSAFE", "SAFE"],
        colorscale=colorscale,
        showscale=True,
        colorbar=dict(
            title=dict(text="Count", font=dict(size=20)),
            tickfont=dict(size=18),
            len=0.6,
            thickness=30
        ),
        hovertemplate=(
            "Actual: %{y}<br>"
            "Predicted: %{x}<br>"
            "Count: %{z}<extra></extra>"
        )
    ))

    # Add text annotations to each cell
    for i in range(2):
        for j in range(2):
            # Determine text color based on background darkness
            text_color = "white" if cm[i, j] > 100 else "black"
            fig.add_annotation(
                x=j,
                y=i,
                text=annotations[i][j],
                showarrow=False,
                font=dict(size=48, color=text_color, family="Arial Black"),
                xref="x",
                yref="y"
            )

    # Add accuracy box below title
    accuracy_text = (
        f"<b>Overall Accuracy: {accuracy:.1%}</b>  |  "
        f"Precision: {precision:.1%}  |  "
        f"Recall: {recall:.1%}  |  "
        f"Total samples: {total}"
    )

    fig.add_annotation(
        text=accuracy_text,
        xref="paper", yref="paper",
        x=0.5, y=1.02,
        xanchor="center", yanchor="bottom",
        showarrow=False,
        font=dict(size=24),
        align="center"
    )

    # Caption text
    caption_text = (
        "<b>Confusion Matrix Interpretation:</b><br>"
        f"<b>True Negatives ({tn})</b>: Model correctly identified UNSAFE water<br>"
        f"<b>True Positives ({tp})</b>: Model correctly identified SAFE water<br>"
        f"<b>False Positives ({fp})</b>: Model said SAFE but water was actually UNSAFE "
        "<i>(most critical error type)</i><br>"
        f"<b>False Negatives ({fn})</b>: Model said UNSAFE but water was actually SAFE<br>"
        "<br>"
        "<b>Model:</b> Random Forest Classifier (class_weight='balanced') | "
        "<b>Training Data:</b> Kaggle Water Quality Dataset (1991-2017)<br>"
        "<b>Features:</b> pH, dissolved oxygen, temperature, nitrate, conductance + temporal features<br>"
        "<br>"
        "<i>Note: Model cannot detect lead, bacteria, PFAS, or other contaminants not in the NSF-WQI methodology.</i>"
    )

    fig.add_annotation(
        text=caption_text,
        xref="paper", yref="paper",
        x=0, y=-0.22,
        xanchor="left", yanchor="top",
        showarrow=False,
        font=dict(size=18),
        align="left"
    )

    # Update layout for poster quality
    fig.update_layout(
        title=dict(
            text="Water Quality Classifier Performance",
            font=dict(size=36, family="Arial, sans-serif"),
            x=0.5,
            xanchor="center",
            y=0.95
        ),
        xaxis=dict(
            title=dict(
                text="Predicted Classification",
                font=dict(size=28, family="Arial, sans-serif")
            ),
            tickfont=dict(size=24),
            side="bottom",
            tickangle=0
        ),
        yaxis=dict(
            title=dict(
                text="Actual Classification",
                font=dict(size=28, family="Arial, sans-serif")
            ),
            tickfont=dict(size=24),
            autorange="reversed"  # Put UNSAFE at top, SAFE at bottom
        ),
        width=2400,
        height=2000,
        margin=dict(l=150, r=150, t=200, b=450),
        plot_bgcolor="white",
        paper_bgcolor="white"
    )

    # Create output directory
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    # Export PNG
    timestamp = datetime.now().strftime("%Y%m%d")
    png_path = output_dir / f"confusion_matrix_poster_{timestamp}.png"

    print(f"\nExporting high-resolution PNG...")
    try:
        fig.write_image(str(png_path), width=2400, height=2000, scale=2)
        file_size = png_path.stat().st_size / (1024 * 1024)
        print(f"  Saved PNG: {png_path} ({file_size:.1f} MB)")
        print(f"  Resolution: 4800 x 4000 pixels (scale=2 for print quality)")
    except Exception as e:
        print(f"\nERROR: Failed to export PNG: {e}")
        print("Make sure kaleido is installed: poetry add kaleido")
        raise SystemExit(1)

    print(f"\nSUCCESS: Confusion matrix chart created.")
    print(f"\nKey insights for poster:")
    print(f"  - 98.3% overall accuracy ({tp + tn}/{total} correct)")
    print(f"  - Only {fp} false positives (said SAFE when UNSAFE)")
    print(f"  - Only {fn} false negatives (said UNSAFE when SAFE)")


if __name__ == "__main__":
    main()
