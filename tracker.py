"""
CorrectorWorkshop — Live Training Tracker
Run with:  .venv/Scripts/streamlit run tracker.py
"""
import time
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml
from PIL import Image

# ── Config ───────────────────────────────────────────────────────────────────
ARTIFACTS_DIR = Path("training_artifacts")
REFRESH_SECS  = 5

# ── Page setup ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CorrectorWorkshop",
    page_icon="⚡",
    layout="wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
[data-testid="stAppViewContainer"] { background: #0f1117; }
[data-testid="stHeader"]           { background: transparent; }
[data-testid="stSidebar"]          { background: #141821; }
[data-testid="stProgress"]         { background: #1e2a3a; }

.metric-card {
    background: #141821;
    border: 1px solid #1e2a3a;
    border-radius: 12px;
    padding: 14px 12px 12px;
    text-align: center;
    min-height: 90px;
}
.metric-label {
    font-size: 10px;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    margin-bottom: 6px;
}
.metric-pre {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: #334155;
    margin-bottom: 1px;
}
.metric-arrow {
    font-size: 11px;
    color: #1e2a3a;
    margin: 0 2px;
}
.metric-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 20px;
    font-weight: 700;
    line-height: 1;
}
.metric-value-lg {
    font-family: 'JetBrains Mono', monospace;
    font-size: 22px;
    font-weight: 700;
    line-height: 1;
    margin-top: 10px;
}
.run-tag {
    display:inline-block;
    background:#1e2a3a;
    border-radius:6px;
    padding:3px 10px;
    font-family:'JetBrains Mono',monospace;
    font-size:12px;
    color:#64748b;
}
.section-label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #334155;
    margin: 24px 0 12px;
}
</style>
""", unsafe_allow_html=True)

# ── Helpers ──────────────────────────────────────────────────────────────────
def find_latest_run() -> Path | None:
    runs = sorted(ARTIFACTS_DIR.glob("train_run_*"))
    return runs[-1] if runs else None


@st.cache_data(ttl=REFRESH_SECS)
def load_csv(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        df["lr"] = df["lr"].astype(float)
        return df
    except Exception:
        return pd.DataFrame()


def latest_image(run_dir: Path) -> Path | None:
    imgs = sorted((run_dir / "samples").glob("sample_*.png"))
    return imgs[-1] if imgs else None


def load_configs(run_dir: Path) -> dict:
    cfg = {}
    for f in (run_dir / "configs").glob("*.yaml"):
        with open(f) as fh:
            cfg[f.stem] = yaml.safe_load(fh)
    return cfg


def dark_layout(**kwargs) -> dict:
    base = dict(
        paper_bgcolor="#141821",
        plot_bgcolor="#141821",
        font=dict(color="#94a3b8", family="Inter", size=11),
        margin=dict(l=50, r=16, t=36, b=28),
        xaxis=dict(gridcolor="#1a2233", zerolinecolor="#1a2233", linecolor="#1a2233"),
        yaxis=dict(gridcolor="#1a2233", zerolinecolor="#1a2233", linecolor="#1a2233"),
        legend=dict(
            orientation="h", y=1.18,
            font=dict(size=10, color="#64748b"),
            bgcolor="rgba(0,0,0,0)",
        ),
    )
    base.update(kwargs)
    return base


def metric_card(label: str, value: str, color: str, pre_value: str | None = None) -> str:
    """Render a metric card. If pre_value given, shows  pre → value  layout."""
    if pre_value is not None:
        body = (
            f"<div class='metric-pre'>{pre_value}</div>"
            f"<div style='font-size:10px;color:#1e3a52;margin:1px 0'>▼</div>"
            f"<div class='metric-value' style='color:{color}'>{value}</div>"
        )
    else:
        body = f"<div class='metric-value-lg' style='color:{color}'>{value}</div>"
    return (
        f"<div class='metric-card'>"
        f"<div class='metric-label'>{label}</div>"
        f"{body}"
        f"</div>"
    )


# ── Data loading ─────────────────────────────────────────────────────────────
run_dir = find_latest_run()

if run_dir is None:
    st.error("No training runs found in `training_artifacts/`")
    st.stop()

df      = load_csv(str(run_dir / "loss.csv"))
configs = load_configs(run_dir)

# Detect whether this run has the new dual-column format
HAS_PRE = "illegal_pair_pct_k1" in df.columns

# Derive total iterations from whichever train config is present
total_iters = 10_000
for cfg in configs.values():
    if isinstance(cfg, dict) and "num_iterations" in cfg:
        total_iters = cfg["num_iterations"]
        break

# ── Header ───────────────────────────────────────────────────────────────────
hcol1, hcol2 = st.columns([5, 2])
with hcol1:
    st.markdown("## ⚡ CorrectorWorkshop — Live Tracker")
with hcol2:
    st.markdown(
        f"<div style='margin-top:18px;text-align:right'>"
        f"<span class='run-tag'>{run_dir.name}</span></div>",
        unsafe_allow_html=True,
    )

if df.empty:
    st.info("Waiting for first metrics…")
    time.sleep(REFRESH_SECS)
    st.rerun()

last = df.iloc[-1]
cur_iter = int(last["iteration"])
progress = cur_iter / total_iters

# Progress bar
st.progress(
    progress,
    text=f"Iteration **{cur_iter:,}** / {total_iters:,}  ·  {progress*100:.1f}% complete"
         f"  ·  {last['lr']:.2e} lr",
)

# ── Metric cards ─────────────────────────────────────────────────────────────
st.markdown("<div class='section-label'>Current metrics</div>", unsafe_allow_html=True)

if HAS_PRE:
    cards = [
        ("Loss",            None,
         f"{last['loss']:.5f}",                                              "#60a5fa"),
        ("Illegal pairs",   f"{last['illegal_pair_pct_pre']:.2f}%",
         f"{last['illegal_pair_pct_k1']:.2f}%",                             "#f87171"),
        ("Viol/cloud",      f"{last['mean_viol_per_cloud_pre']:.2f}",
         f"{last['mean_viol_per_cloud_k1']:.2f}",                           "#4ade80"),
        ("Viol reduction",  None,
         f"{last['viol_reduction_pct_k1']:.1f}%",                           "#4ade80"),
        ("Displacement",    None,
         f"{last['displacement_rel_rd_k1']:.3f}x rd",                       "#fb923c"),
        ("Mean NN dist",    f"{last['mean_nn_dist_pre']:.4f}",
         f"{last['mean_nn_dist_k1']:.4f}",                                  "#14b8a6"),
    ]
else:
    cards = [
        ("Loss",           None, f"{last['loss']:.5f}",                    "#60a5fa"),
        ("Illegal pairs",  None, f"{last['illegal_pair_pct']:.2f}%",       "#f87171"),
        ("Viol/cloud",     None, f"{last.get('mean_viol_per_cloud_k1', 0):.2f}", "#4ade80"),
        ("Viol reduction", None, f"{last.get('viol_reduction_pct_k1', 0):.1f}%", "#4ade80"),
        ("Displacement",   None, f"{last.get('displacement_rel_rd_k1', last.get('displacement_rel_rd', 0)):.3f}x rd", "#fb923c"),
        ("Mean NN dist",   None, f"{last.get('mean_nn_dist_k1', last.get('mean_nn_dist', 0)):.4f}", "#14b8a6"),
    ]

for col, (label, pre, value, color) in zip(st.columns(6), cards):
    with col:
        st.markdown(metric_card(label, value, color, pre), unsafe_allow_html=True)

st.markdown("<div style='margin:8px 0'></div>", unsafe_allow_html=True)

# ── Charts + image ───────────────────────────────────────────────────────────
chart_col, img_col = st.columns([3, 2], gap="large")

with chart_col:
    st.markdown("<div class='section-label'>Training curves</div>", unsafe_allow_html=True)

    # 1. Loss
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["iteration"], y=df["loss"], mode="lines", name="loss",
        line=dict(color="#60a5fa", width=2),
        fill="tozeroy", fillcolor="rgba(96,165,250,0.07)",
    ))
    fig.update_layout(**dark_layout(title="Loss", height=200, showlegend=False))
    st.plotly_chart(fig, use_container_width=True)

    # 2a. Illegal pairs %
    fig = go.Figure()
    if HAS_PRE:
        fig.add_trace(go.Scatter(
            x=df["iteration"], y=df["illegal_pair_pct_pre"], mode="lines",
            name="before", opacity=0.35,
            line=dict(color="#f87171", width=1.2, dash="dot"),
        ))
    col_ill = "illegal_pair_pct_k1" if HAS_PRE else "illegal_pair_pct"
    fig.add_trace(go.Scatter(
        x=df["iteration"], y=df[col_ill], mode="lines",
        name="after (K=1)", line=dict(color="#f87171", width=2),
        fill="tozeroy", fillcolor="rgba(248,113,113,0.07)",
    ))
    fig.update_layout(**dark_layout(
        title="Illegal Pairs %  (dotted = before · target: 0%)",
        height=180,
        yaxis=dict(gridcolor="#1a2233", color="#f87171", title=None, rangemode="tozero"),
    ))
    st.plotly_chart(fig, use_container_width=True)

    # 2b. Violations per cloud + reduction rate
    fig = go.Figure()
    if HAS_PRE and "mean_viol_per_cloud_pre" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["iteration"], y=df["mean_viol_per_cloud_pre"], mode="lines",
            name="viol/cloud (before)", opacity=0.35,
            line=dict(color="#4ade80", width=1.2, dash="dot"),
        ))
        fig.add_trace(go.Scatter(
            x=df["iteration"], y=df["mean_viol_per_cloud_k1"], mode="lines",
            name="viol/cloud (K=1)", line=dict(color="#4ade80", width=2),
            fill="tozeroy", fillcolor="rgba(74,222,128,0.07)",
        ))
        fig.add_trace(go.Scatter(
            x=df["iteration"], y=df["viol_reduction_pct_k1"], mode="lines",
            name="reduction % (K=1)", line=dict(color="#a3e635", width=1.5, dash="dash"),
            yaxis="y2",
        ))
    fig.update_layout(**dark_layout(
        title="Violations per cloud  &  Reduction %  (dashed, right axis)",
        height=180,
        yaxis=dict(gridcolor="#1a2233", color="#4ade80", title=None, rangemode="tozero"),
        yaxis2=dict(overlaying="y", side="right", color="#a3e635",
                    gridcolor="rgba(0,0,0,0)", title=None, range=[0, 100]),
    ))
    st.plotly_chart(fig, use_container_width=True)

    # 3. Mean violation pre→post  (only when new format)
    if HAS_PRE and "mean_violation_k1" in df.columns:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["iteration"], y=df["mean_violation_pre"], mode="lines",
            name="mean violation (before)", opacity=0.4,
            line=dict(color="#f472b6", width=1.2, dash="dot"),
        ))
        fig.add_trace(go.Scatter(
            x=df["iteration"], y=df["mean_violation_k1"], mode="lines",
            name="mean violation (K=1)", line=dict(color="#f472b6", width=2),
            fill="tozeroy", fillcolor="rgba(244,114,182,0.06)",
        ))
        if "median_violation_k1" in df.columns:
            fig.add_trace(go.Scatter(
                x=df["iteration"], y=df["median_violation_k1"], mode="lines",
                name="median violation (K=1)", opacity=0.6,
                line=dict(color="#f472b6", width=1.2, dash="dash"),
            ))
        fig.update_layout(**dark_layout(
            title="Mean Violation  (dotted = before, dashed = median after)",
            height=180,
        ))
        st.plotly_chart(fig, use_container_width=True)

    # 4. Displacement & NN dist
    fig = go.Figure()
    col_disp     = "displacement_rel_rd_k1"     if HAS_PRE else "displacement_rel_rd"
    col_disp_med = "displacement_rel_rd_median_k1" if HAS_PRE else "displacement_rel_rd_median"
    col_nn       = "mean_nn_dist_k1"             if HAS_PRE else "mean_nn_dist"
    fig.add_trace(go.Scatter(
        x=df["iteration"], y=df[col_disp], mode="lines",
        name="displacement (x rd)", line=dict(color="#fb923c", width=2),
    ))
    if col_disp_med in df.columns:
        fig.add_trace(go.Scatter(
            x=df["iteration"], y=df[col_disp_med], mode="lines",
            name="displacement median (x rd)", opacity=0.5,
            line=dict(color="#fb923c", width=1.2, dash="dash"),
        ))
    if HAS_PRE and "mean_nn_dist_pre" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["iteration"], y=df["mean_nn_dist_pre"], mode="lines",
            name="mean NN dist (before)", opacity=0.35,
            line=dict(color="#14b8a6", width=1.2, dash="dot"),
            yaxis="y2",
        ))
    fig.add_trace(go.Scatter(
        x=df["iteration"], y=df[col_nn], mode="lines",
        name="mean NN dist (after)", line=dict(color="#14b8a6", width=2),
        yaxis="y2",
    ))
    fig.add_hline(y=1.0, line_dash="dash", line_color="#fb923c",
                  opacity=0.3, annotation_text="1x rd", annotation_font_size=10)
    fig.update_layout(**dark_layout(
        title="Displacement  &  Nearest-Neighbour Distance  (dotted = before)",
        height=200,
        yaxis=dict(gridcolor="#1a2233", color="#fb923c", title=None),
        yaxis2=dict(overlaying="y", side="right", color="#14b8a6",
                    gridcolor="rgba(0,0,0,0)", title=None),
    ))
    st.plotly_chart(fig, use_container_width=True)

    # 5. Learning rate
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["iteration"], y=df["lr"], mode="lines", name="lr",
        line=dict(color="#a78bfa", width=2),
        fill="tozeroy", fillcolor="rgba(167,139,250,0.07)",
    ))
    fig.update_layout(**dark_layout(
        title="Learning Rate", height=160,
        showlegend=False,
        xaxis=dict(gridcolor="#1a2233", title="iteration"),
    ))
    st.plotly_chart(fig, use_container_width=True)

with img_col:
    st.markdown("<div class='section-label'>Latest sample</div>", unsafe_allow_html=True)
    img_path = latest_image(run_dir)
    if img_path:
        st.markdown(
            f"<p style='color:#334155;font-size:11px;margin-bottom:8px;font-family:monospace'>"
            f"{img_path.name}</p>",
            unsafe_allow_html=True,
        )
        st.image(Image.open(img_path), use_container_width=True)

        # Show config summary
        st.markdown("<div class='section-label' style='margin-top:24px'>Run config</div>",
                    unsafe_allow_html=True)
        for fname, cfg in configs.items():
            if not isinstance(cfg, dict):
                continue
            with st.expander(fname, expanded=False):
                st.json(cfg)
    else:
        st.markdown("""
        <div style='background:#141821;border:1px solid #1e2a3a;border-radius:12px;
                    padding:60px 24px;text-align:center;color:#334155;margin-top:8px;
                    font-size:13px'>
            ⏳ Waiting for first sample image…
        </div>""", unsafe_allow_html=True)

# ── Footer + auto-refresh ────────────────────────────────────────────────────
st.markdown(
    f"<p style='color:#1e2a3a;font-size:11px;text-align:center;margin-top:24px'>"
    f"auto-refreshing every {REFRESH_SECS}s</p>",
    unsafe_allow_html=True,
)
time.sleep(REFRESH_SECS)
st.rerun()
