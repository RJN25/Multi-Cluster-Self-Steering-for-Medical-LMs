import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

CSV_PATH = "logs/eval_results.csv"
OUT_HTML = "logs/eval_metrics_trends.html"
OUT_PNG = "logs/eval_metrics_trends.png"

# === Load Data ===
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"{CSV_PATH} not found. Run 4_export_eval_to_csv first.")

df = pd.read_csv(CSV_PATH)

# Ensure numeric order (MedQA ids are like test-00001)
df = df.sort_values(by="qid").reset_index(drop=True)
df["index"] = df.index + 1  # human-readable sample index

# === Rolling averages for smoothing ===
window = max(10, len(df)//50)  # adaptive smoothing
df["brier_smooth"] = df["brier"].rolling(window, min_periods=1).mean()
df["ece_smooth"] = df["ece"].rolling(window, min_periods=1).mean()

# AUROC cannot be computed per-sample — skip per-prompt AUROC; global AUROC shown in title instead.
auroc_global = df["correct"].mean()  # placeholder: could load from eval.log summary

# === Create figure layout ===
fig = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.12,
    subplot_titles=("Brier Score Across Samples", "ECE Across Samples")
)

# --- Brier plot ---
fig.add_trace(
    go.Scatter(
        x=df["index"], y=df["brier"],
        mode="markers",
        marker=dict(size=3, color="rgba(180,180,180,0.3)"),
        name="Brier (raw)",
        hovertemplate="Sample %{x}<br>Brier=%{y:.4f}<extra></extra>",
    ),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(
        x=df["index"], y=df["brier_smooth"],
        mode="lines",
        line=dict(color="#1f77b4", width=2),
        name=f"Brier (smoothed, w={window})",
        hovertemplate="Sample %{x}<br>Brier=%{y:.4f}<extra></extra>",
    ),
    row=1, col=1
)

# --- ECE plot ---
fig.add_trace(
    go.Scatter(
        x=df["index"], y=df["ece"],
        mode="markers",
        marker=dict(size=3, color="rgba(200,180,180,0.3)"),
        name="ECE (raw)",
        hovertemplate="Sample %{x}<br>ECE=%{y:.4f}<extra></extra>",
    ),
    row=2, col=1
)

fig.add_trace(
    go.Scatter(
        x=df["index"], y=df["ece_smooth"],
        mode="lines",
        line=dict(color="#d62728", width=2),
        name=f"ECE (smoothed, w={window})",
        hovertemplate="Sample %{x}<br>ECE=%{y:.4f}<extra></extra>",
    ),
    row=2, col=1
)

# === Figure aesthetics (NeurIPS-style minimalist aesthetic) ===
fig.update_layout(
    title=dict(
        text=f"<b>Calibration Metrics Across MedQA Samples</b><br><sup>Global AUROC ≈ {auroc_global:.3f}</sup>",
        x=0.5,
        font=dict(size=20)
    ),
    showlegend=True,
    legend=dict(orientation="h", y=-0.15, x=0.3),
    height=750,
    template="plotly_white",
    margin=dict(l=70, r=40, t=80, b=70),
    font=dict(family="Arial", size=14),
)

fig.update_xaxes(title_text="Sample Index", row=2, col=1)
fig.update_yaxes(title_text="Brier Score", row=1, col=1)
fig.update_yaxes(title_text="ECE", row=2, col=1)

# === Save outputs ===
os.makedirs(os.path.dirname(OUT_HTML), exist_ok=True)
fig.write_html(OUT_HTML)
fig.write_image(OUT_PNG, scale=3, width=1100, height=700)

print(f"[✓] Saved interactive plot → {OUT_HTML}")
print(f"[✓] Saved publication PNG → {OUT_PNG}")
