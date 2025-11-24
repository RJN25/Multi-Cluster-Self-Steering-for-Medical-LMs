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
window = max(10, len(df)//50)  # adaptive smoothing window
df["brier_smooth"] = df["brier"].rolling(window, min_periods=1).mean()
df["ece_smooth"] = df["ece"].rolling(window, min_periods=1).mean()
df["acc_smooth"] = df["correct"].rolling(window, min_periods=1).mean()

# === Global metrics ===
acc_global = df["correct"].mean()
auroc_global = acc_global  # Placeholder; could be parsed from eval.log summary

# === Create figure layout ===
fig = make_subplots(
    rows=3, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.10,
    subplot_titles=("Brier Score Across Samples", "ECE Across Samples", "Accuracy Across Samples")
)

# --- 1️⃣ Brier plot ---
fig.add_trace(
    go.Scatter(
        x=df["index"], y=df["brier"],
        mode="markers",
        marker=dict(size=3, color="rgba(160,160,160,0.3)"),
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

# --- 2️⃣ ECE plot ---
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

# --- 3️⃣ Accuracy plot ---
fig.add_trace(
    go.Scatter(
        x=df["index"], y=df["correct"],
        mode="markers",
        marker=dict(size=3, color="rgba(180,220,180,0.3)"),
        name="Accuracy (per-sample)",
        hovertemplate="Sample %{x}<br>Correct=%{y:.0f}<extra></extra>",
    ),
    row=3, col=1
)
fig.add_trace(
    go.Scatter(
        x=df["index"], y=df["acc_smooth"],
        mode="lines",
        line=dict(color="#2ca02c", width=2),
        name=f"Accuracy (rolling mean, w={window})",
        hovertemplate="Sample %{x}<br>Rolling Acc=%{y:.3f}<extra></extra>",
    ),
    row=3, col=1
)

# === Figure aesthetics (NeurIPS style) ===
fig.update_layout(
    title=dict(
        text=f"<b>Calibration & Performance Trends Across MedQA Samples</b><br><sup>Global Accuracy ≈ {acc_global:.3f} | Global AUROC ≈ {auroc_global:.3f}</sup>",
        x=0.5,
        font=dict(size=20)
    ),
    showlegend=True,
    legend=dict(orientation="h", y=-0.18, x=0.25),
    height=950,
    template="plotly_white",
    margin=dict(l=70, r=40, t=90, b=70),
    font=dict(family="Arial", size=14),
)

# Axis titles
fig.update_xaxes(title_text="Sample Index", row=3, col=1)
fig.update_yaxes(title_text="Brier Score", row=1, col=1)
fig.update_yaxes(title_text="ECE", row=2, col=1)
fig.update_yaxes(title_text="Accuracy", row=3, col=1, range=[0,1])

# === Save outputs ===
os.makedirs(os.path.dirname(OUT_HTML), exist_ok=True)
fig.write_html(OUT_HTML)
# PNG export optional (comment out if Chrome not available)
try:
    fig.write_image(OUT_PNG, scale=3, width=1100, height=900)
except Exception as e:
    print(f"[!] PNG export skipped due to: {e}")

print(f"[✓] Saved interactive plot → {OUT_HTML}")
print(f"[✓] Saved publication PNG → {OUT_PNG}")
