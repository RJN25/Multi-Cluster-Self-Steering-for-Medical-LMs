import os
import re
import plotly.graph_objects as go

# === File paths ===
LOG_DIR = "graph_logs"
DSAC_LOG = os.path.join(LOG_DIR, "dsac1.log")
QWEN_LOG = os.path.join(LOG_DIR, "qwen2.log")
OUT_HTML = os.path.join(LOG_DIR, "comparison_metrics.html")
OUT_PNG = os.path.join(LOG_DIR, "comparison_metrics.png")

# === Regex to capture final summary lines ===
summary_re = re.compile(
    r"(?:MEAN_CONFIDENCE|MeanConf)=(?P<conf>[\d.]+).*?"
    r"ACCURACY=(?P<acc>[\d.]+).*?"
    r"AUROC=(?P<auroc>[\d.]+).*?"
    r"Brier=(?P<brier>[\d.]+).*?"
    r"ECE=(?P<ece>[\d.]+)",
    re.IGNORECASE
)

def extract_metrics(path):
    text = open(path).read()
    m = summary_re.search(text)
    if not m:
        raise ValueError(f"Could not parse summary from {path}")
    return {k: float(v) for k, v in m.groupdict().items()}

dsac = extract_metrics(DSAC_LOG)
qwen = extract_metrics(QWEN_LOG)

# === Data setup ===
metrics = ["Accuracy", "Mean Confidence", "AUROC", "Brier", "ECE"]
dsac_vals = [dsac["acc"], dsac["conf"], dsac["auroc"], dsac["brier"], dsac["ece"]]
qwen_vals = [qwen["acc"], qwen["conf"], qwen["auroc"], qwen["brier"], qwen["ece"]]

# Whether higher or lower is better
direction = ["↑", "—", "↑", "↓", "↓"]

# Colors and layout
colors = {"DSAC": "#1f77b4", "Qwen": "rgba(255,100,100,0.7)"}

# === Bar plot ===
fig = go.Figure()

fig.add_trace(go.Bar(
    x=metrics, y=qwen_vals,
    name="Qwen-2.5-3B (Baseline)",
    marker_color=colors["Qwen"],
    text=[f"{v:.4f}" for v in qwen_vals],
    textposition="outside",
))

fig.add_trace(go.Bar(
    x=metrics, y=dsac_vals,
    name="DSAC (Ours)",
    marker_color=colors["DSAC"],
    text=[f"{v:.4f}" for v in dsac_vals],
    textposition="outside",
))

# === Aesthetics (NeurIPS-style) ===
fig.update_layout(
    title=dict(
        text="<b>Confidence Calibration & Accuracy Comparison</b><br>"
             "<sup>MedQA-USMLE Evaluation (DSAC vs. Qwen-2.5-3B Baseline)</sup>",
        x=0.5, font=dict(size=20)
    ),
    barmode="group",
    bargap=0.25,
    template="plotly_white",
    font=dict(family="Arial", size=14),
    height=600,
    margin=dict(l=70, r=40, t=90, b=80),
    legend=dict(
        orientation="h", y=-0.15, x=0.25,
        bgcolor="rgba(0,0,0,0)", font=dict(size=13)
    )
)

# Add up/down arrows below each metric label
fig.update_xaxes(
    tickvals=list(range(len(metrics))),
    ticktext=[f"{m}<br><span style='font-size:12px;color:gray'>{d}</span>"
              for m, d in zip(metrics, direction)]
)

# === Export ===
os.makedirs(LOG_DIR, exist_ok=True)
fig.write_html(OUT_HTML)
try:
    fig.write_image(OUT_PNG, scale=3, width=1000, height=600)
except Exception as e:
    print(f"[!] PNG export skipped due to: {e}")

print(f"[✓] Saved interactive plot → {OUT_HTML}")
print(f"[✓] Saved publication PNG → {OUT_PNG}")
