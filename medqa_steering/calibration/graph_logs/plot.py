import plotly.graph_objects as go
import os

# === Metric values ===
dsac = dict(
    Accuracy=0.2396,
    Mean_Confidence=0.3068,
    AUROC=0.6816,
    Brier=0.7468,
    ECE=0.0656
)

qwen = dict(
    Accuracy=0.2476,
    Mean_Confidence=0.4757,
    AUROC=0.6917,
    Brier=0.7925,
    ECE=0.1358
)

# === Define which metrics are “higher is better” ===
better_up = {"Accuracy", "AUROC", "Mean_Confidence"}
better_down = {"Brier", "ECE"}

# === Arrange metrics in a consistent order ===
metrics = ["Accuracy", "Mean_Confidence", "AUROC", "Brier", "ECE"]

# === Build figure ===
dsac_vals = [dsac[m] for m in metrics]
qwen_vals = [qwen[m] for m in metrics]

# Choose colors with subtle contrast
colors = {
    "DSAC": "rgba(31, 119, 180, 0.8)",   # blue
    "Qwen": "rgba(214, 39, 40, 0.8)"     # red
}

fig = go.Figure()

# DSAC bars
fig.add_trace(go.Bar(
    x=metrics,
    y=dsac_vals,
    name="DSAC (Ours)",
    marker_color=colors["DSAC"],
    text=[f"{v:.3f}" for v in dsac_vals],
    textposition="outside"
))

# Qwen bars
fig.add_trace(go.Bar(
    x=metrics,
    y=qwen_vals,
    name="Qwen Baseline",
    marker_color=colors["Qwen"],
    text=[f"{v:.3f}" for v in qwen_vals],
    textposition="outside"
))

# === Add arrows to indicate improvement direction ===
annotations = []
for i, m in enumerate(metrics):
    direction = "↑" if m in better_up else "↓"
    annotations.append(dict(
        x=m, y=max(dsac_vals[i], qwen_vals[i]) * 1.05,
        text=f"<b>{direction}</b>",
        showarrow=False,
        font=dict(size=18, color="gray")
    ))

# === Layout aesthetics (NeurIPS style) ===
fig.update_layout(
    title=dict(
        text="<b>DSAC vs Qwen Baseline: Calibration and Confidence Metrics</b><br><sup>Higher ↑ or Lower ↓ Indicates Better Performance</sup>",
        x=0.5,
        font=dict(size=20)
    ),
    barmode="group",
    bargap=0.25,
    bargroupgap=0.05,
    annotations=annotations,
    template="plotly_white",
    legend=dict(
        orientation="h", y=-0.2, x=0.25,
        font=dict(size=14)
    ),
    font=dict(family="Arial", size=14),
    margin=dict(l=60, r=40, t=90, b=80),
    height=650
)

# Add axis labels
fig.update_yaxes(title_text="Metric Value", showgrid=True, gridcolor="lightgray")
fig.update_xaxes(title_text="")

# === Export ===
os.makedirs("graph_logs", exist_ok=True)
fig.write_html("graph_logs/dsac_qwen_comparison.html")
try:
    fig.write_image("graph_logs/dsac_qwen_comparison.png", scale=3, width=1100, height=700)
except Exception as e:
    print("[!] PNG export skipped due to:", e)

print("[✓] Saved interactive plot → graph_logs/dsac_qwen_comparison.html")
print("[✓] Saved publication PNG → graph_logs/dsac_qwen_comparison.png")
