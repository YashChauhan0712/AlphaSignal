import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.metrics import confusion_matrix

from stock_model import run_single_ticker


# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AlphaSignal · ML Backtesting",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

/* Root palette */
:root {
    --bg:        #080d13;
    --surface:   #0e1620;
    --border:    #233040;
    --accent:    #00e5a0;
    --accent2:   #60a5fa;
    --danger:    #fb7185;
    --muted:     #64849a;
    --text:      #eaf4ff;
    --subtext:   #a8c4d8;
    --label:     #7fb3cc;
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif;
    font-size: 15px;
    line-height: 1.65;
}

/* Paragraphs & markdown body */
p, li, [data-testid="stMarkdown"] p {
    color: var(--subtext) !important;
    line-height: 1.7;
}

/* Bold text inside markdown should pop */
strong { color: var(--text) !important; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
    padding-top: 8px;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }
[data-testid="stSidebar"] p { color: var(--subtext) !important; }

/* Sidebar section spacing */
[data-testid="stSidebar"] .stMarkdown { margin-bottom: 4px; }
[data-testid="stSidebar"] .stTextInput,
[data-testid="stSidebar"] .stSlider,
[data-testid="stSidebar"] .stNumberInput { margin-bottom: 12px; }

/* Sidebar inputs */
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] select,
[data-testid="stSidebar"] textarea {
    background: #111c28 !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 4px !important;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 16px 20px;
}
[data-testid="stMetricLabel"] {
    color: var(--label) !important;
    font-size: 0.7rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 4px;
}
[data-testid="stMetricValue"] {
    color: var(--text) !important;
    font-family: 'Space Mono', monospace;
    font-size: 1.5rem;
    font-weight: 700;
    line-height: 1.2;
}
[data-testid="stMetricDelta"] { color: var(--label) !important; font-size: 0.78rem; }

/* Dataframes */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: 6px;
    overflow: hidden;
}

/* Buttons */
button[kind="primary"], .stButton > button {
    background: var(--accent) !important;
    color: #000000 !important;
    font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 4px !important;
    letter-spacing: 0.06em;
    padding: 10px 0 !important;
    transition: opacity 0.15s;
    -webkit-text-fill-color: #000000 !important;
}
.stButton > button:hover { opacity: 0.85 !important; }
.stButton > button *, .stButton > button p {
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
}

/* Headers */
h1, h2, h3, h4 { font-family: 'Space Mono', monospace !important; }
h1 { font-size: 1.6rem !important; color: var(--text) !important; letter-spacing: -0.01em; margin-bottom: 2px !important; }
h2 { font-size: 1.0rem !important; color: var(--subtext) !important; font-weight: 400 !important; margin-top: 2px !important; }
h3 { font-size: 1.05rem !important; color: var(--text) !important; margin-bottom: 10px !important; }
h4 { font-size: 0.85rem !important; color: var(--label) !important; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 10px !important; font-weight: 600 !important; }

/* Dividers */
hr { border-color: var(--border) !important; margin: 20px 0 !important; }

/* Spinner */
[data-testid="stSpinner"] { color: var(--accent) !important; }

/* ── Force the entire main content area dark ── */
[data-testid="stMain"],
[data-testid="stMainBlockContainer"],
section[data-testid="stMainBlockContainer"] > div,
.main > div { background: var(--bg) !important; }

/* Expander — kill the white card */
[data-testid="stExpander"] {
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    background: var(--surface) !important;
    margin-top: 16px;
}
[data-testid="stExpander"] > div,
[data-testid="stExpander"] > div > div,
details[data-testid="stExpander"] {
    background: var(--surface) !important;
    border-radius: 6px !important;
}
[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary * {
    color: var(--subtext) !important;
    background: var(--surface) !important;
}
[data-testid="stExpander"] summary:hover { background: #131f2e !important; }

/* Tabs — kill white tab bar background */
[data-testid="stTabs"] > div:first-child,
[role="tablist"] {
    background: transparent !important;
    border-bottom: 1px solid var(--border) !important;
}
[role="tab"] {
    background: transparent !important;
    color: var(--subtext) !important;
}
[role="tab"][aria-selected="true"] {
    color: var(--text) !important;
    border-bottom: 2px solid var(--accent) !important;
}
[role="tabpanel"] { background: transparent !important; padding-top: 16px; }

/* Dataframe / table — force dark backgrounds */
[data-testid="stDataFrame"] iframe,
[data-testid="stDataFrame"] > div {
    background: var(--surface) !important;
    color: var(--text) !important;
}

/* Download button — match dark theme */
[data-testid="stDownloadButton"] button {
    background: transparent !important;
    border: 1px solid var(--border) !important;
    color: var(--subtext) !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.05em;
    border-radius: 4px !important;
    transition: border-color 0.15s, color 0.15s;
}
[data-testid="stDownloadButton"] button:hover {
    border-color: var(--accent) !important;
    color: var(--accent) !important;
    background: rgba(0,229,160,0.06) !important;
}

/* Caption */
[data-testid="stCaptionContainer"] { color: var(--muted) !important; font-size: 0.78rem; margin-top: 6px; }

/* Regime badge */
.regime-badge {
    display: inline-block;
    padding: 4px 13px;
    border-radius: 3px;
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    vertical-align: middle;
}
.regime-trending { background: rgba(0,229,160,0.15); color: #00e5a0; border: 1px solid rgba(0,229,160,0.35); }
.regime-volatile { background: rgba(251,113,133,0.15); color: #fb7185; border: 1px solid rgba(251,113,133,0.35); }
.regime-mixed    { background: rgba(96,165,250,0.15); color: #60a5fa; border: 1px solid rgba(96,165,250,0.35); }

/* Regime description text */
.regime-desc {
    color: var(--subtext);
    font-size: 0.875rem;
    vertical-align: middle;
}

/* Signal pill */
.signal-up   { background: rgba(0,229,160,0.12); color: #00e5a0; border: 1px solid rgba(0,229,160,0.4); padding: 5px 16px; border-radius: 3px; font-family: 'Space Mono', monospace; font-weight: 700; font-size: 0.85rem; vertical-align: middle; }
.signal-down { background: rgba(251,113,133,0.12); color: #fb7185; border: 1px solid rgba(251,113,133,0.4); padding: 5px 16px; border-radius: 3px; font-family: 'Space Mono', monospace; font-weight: 700; font-size: 0.85rem; vertical-align: middle; }

/* Inline label */
.inline-label { color: var(--label); font-size: 0.82rem; vertical-align: middle; }

/* Section spacer */
.spacer-sm { margin-top: 12px; }
.spacer-md { margin-top: 24px; }

/* Tab overrides */
[data-testid="stTab"] { color: var(--subtext) !important; font-size: 0.9rem; }
[data-testid="stTab"][aria-selected="true"] { color: var(--text) !important; }
</style>
""", unsafe_allow_html=True)


# ── Matplotlib dark theme ─────────────────────────────────────────────────────

def apply_chart_style(fig, ax):
    fig.patch.set_facecolor("#0e1620")
    ax.set_facecolor("#0e1620")
    ax.tick_params(colors="#a8c4d8", labelsize=8.5)
    ax.xaxis.label.set_color("#a8c4d8")
    ax.yaxis.label.set_color("#a8c4d8")
    ax.title.set_color("#eaf4ff")
    ax.title.set_fontsize(10)
    for spine in ax.spines.values():
        spine.set_edgecolor("#233040")
    ax.tick_params(axis="both", which="both", length=0)
    ax.grid(axis="y", color="#1a2a38", linewidth=0.5, linestyle="--")
    return fig, ax


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.markdown("---")

    ticker_input = st.text_input(
        "Tickers (comma-separated)",
        value="AAPL, MSFT, NVDA",
        help="e.g. AAPL, TSLA, ^GSPC"
    )

    st.markdown("**Threshold Mode**")
    threshold_mode = st.radio(
        "",
        ["Adaptive", "Manual"],
        label_visibility="collapsed"
    )

    manual_threshold = st.slider(
        "Decision threshold",
        min_value=0.30,
        max_value=0.70,
        value=0.50,
        step=0.01,
        disabled=(threshold_mode == "Adaptive")
    )

    st.markdown("**Backtest Parameters**")
    start = st.number_input("Start row", min_value=500, max_value=10000, value=2500, step=100)
    step  = st.number_input("Step size", min_value=50,  max_value=1000,  value=250,  step=50)

    st.markdown("---")
    run_button = st.button("▶  RUN ANALYSIS", use_container_width=True)

    st.markdown("""
    <div style='margin-top:28px; color:#7fb3cc; font-size:0.75rem; line-height:1.9; border-top: 1px solid #233040; padding-top: 16px;'>
    Models: Random Forest · XGBoost<br>Logistic Regression · Ensemble<br>
    <span style='color:#64849a;'>Strategy: Walk-forward backtesting<br>with Sharpe &amp; drawdown evaluation</span>
    </div>
    """, unsafe_allow_html=True)


# ── Header ────────────────────────────────────────────────────────────────────

col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.markdown("# AlphaSignal")
    st.markdown(
        "<p style='color:#a8c4d8; font-size:0.95rem; margin-top:-6px; margin-bottom:0;'>"
        "ML-driven stock direction prediction &amp; walk-forward backtesting</p>",
        unsafe_allow_html=True
    )
st.markdown("---")


# ── Helper: regime badge ──────────────────────────────────────────────────────

def regime_badge(regime: str) -> str:
    cls = {
        "Trending": "regime-trending",
        "Volatile": "regime-volatile",
    }.get(regime, "regime-mixed")
    return f'<span class="regime-badge {cls}">{regime.upper()}</span>'


# ── Section renderers ─────────────────────────────────────────────────────────

def display_regime_info(result):
    regime    = result["regime_info"]
    settings  = result["adaptive_settings"]

    r_col1, r_col2, r_col3, r_col4 = st.columns(4)
    r_col1.metric("1Y Return",   f"{regime['total_return_1y']:.2%}")
    r_col2.metric("Volatility",  f"{regime['volatility']:.2%}")
    r_col3.metric("Trend Days",  f"{regime['trend_days']:.2%}")
    r_col4.metric("Threshold",   f"{settings['threshold']:.2f}")

    st.markdown(
        f"<div style='margin-top:14px; margin-bottom:4px;'>"
        f"<span style='color:#7fb3cc; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.1em; font-weight:600;'>REGIME</span>"
        f"&nbsp;&nbsp;{regime_badge(regime['regime'])}"
        f"&nbsp;&nbsp;<span class='regime-desc'>{regime['description']}</span>"
        f"</div>",
        unsafe_allow_html=True
    )


def display_main_metrics(metrics, strategy_results, result):
    st.markdown(
        "<p style='color:#7fb3cc; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.1em; font-weight:600; margin-bottom:8px;'>CLASSIFICATION METRICS</p>",
        unsafe_allow_html=True
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy",  f"{metrics['accuracy']:.3f}")
    c2.metric("Precision", f"{metrics['precision']:.3f}")
    c3.metric("Recall",    f"{metrics['recall']:.3f}")
    c4.metric("F1 Score",  f"{metrics['f1']:.3f}")

    st.markdown(
        "<p style='color:#7fb3cc; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.1em; font-weight:600; margin-top:20px; margin-bottom:8px;'>STRATEGY PERFORMANCE</p>",
        unsafe_allow_html=True
    )
    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Ensemble Sharpe",   f"{strategy_results['strategy_sharpe']:.3f}")
    c6.metric("Buy & Hold Sharpe", f"{strategy_results['buy_hold_sharpe']:.3f}")
    c7.metric("Win Rate",          f"{strategy_results['win_rate']:.2%}")
    c8.metric("# Trades",          f"{strategy_results['number_of_trades']:,}")

    signal_cls = "signal-up" if result["latest_signal"] == 1 else "signal-down"
    signal_txt = "▲ UP" if result["latest_signal"] == 1 else "▼ DOWN"
    st.markdown(
        f"<div style='margin-top:18px; display:flex; align-items:center; gap:12px;'>"
        f"<span style='color:#7fb3cc; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.1em; font-weight:600;'>LATEST SIGNAL</span>"
        f'<span class="{signal_cls}">{signal_txt}</span>'
        f"<span style='color:#a8c4d8; font-size:0.875rem;'>P(up) = <strong style='color:#eaf4ff;'>{result['latest_prob']:.3f}</strong></span>"
        f"</div>",
        unsafe_allow_html=True
    )


def display_model_comparison(result):
    tab1, tab2 = st.tabs(["Classification", "Strategy"])

    with tab1:
        df = result["all_model_metrics"].copy()
        def hl_max(s):
            return ["background: rgba(0,229,160,0.25); color: #00e5a0; font-weight:600;" if v == s.max() else "" for v in s]
        styled = df.style.format({"Accuracy": "{:.3f}", "Precision": "{:.3f}", "Recall": "{:.3f}", "F1": "{:.3f}"})
        for col in ["Accuracy", "Precision", "Recall", "F1"]:
            styled = styled.apply(hl_max, subset=[col])
        st.dataframe(styled, use_container_width=True)

    with tab2:
        df = result["strategy_by_model"].sort_values("StrategySharpe", ascending=False).copy()
        fmt = {c: "{:.3f}" for c in df.select_dtypes("float").columns}
        fmt["Trades"] = "{:.0f}"
        def hl_sharpe(s):
            return ["background: rgba(0,229,160,0.25); color: #00e5a0; font-weight:600;" if v == s.max() else "" for v in s]
        styled2 = df.style.format(fmt).apply(hl_sharpe, subset=["StrategySharpe"])
        st.dataframe(styled2, use_container_width=True)


def display_strategy_chart(strategy_results):
    st.markdown(
        "<p style='color:#7fb3cc; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.1em; font-weight:600; margin-bottom:10px;'>CUMULATIVE RETURNS</p>",
        unsafe_allow_html=True
    )

    table          = strategy_results["strategy_table"].copy()
    strategy_curve = (1 + table["Strategy_Return"]).cumprod()
    position_curve = (1 + table["Position_Size_Return"]).cumprod()
    bh_curve       = (1 + table["Buy_Hold_Return"]).cumprod()

    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(bh_curve.index,       bh_curve.values,       color="#64849a", linewidth=1.2, label="Buy & Hold")
    ax.plot(strategy_curve.index, strategy_curve.values, color="#00e5a0", linewidth=2.0, label="Ensemble Strategy")
    ax.plot(position_curve.index, position_curve.values, color="#60a5fa", linewidth=1.4, linestyle="--", label="Position-Sized")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}×"))
    ax.legend(fontsize=8.5, framealpha=0, labelcolor="#eaf4ff")
    fig, ax = apply_chart_style(fig, ax)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

    # Draw-down & stats row
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Strategy Total Return",   f"{strategy_results['strategy_total_return']:.2f}×")
    d2.metric("Buy & Hold Total Return", f"{strategy_results['buy_hold_total_return']:.2f}×")
    d3.metric("Max Drawdown (Strategy)", f"{strategy_results['strategy_max_drawdown']:.2%}")
    d4.metric("Max Drawdown (B&H)",      f"{strategy_results['buy_hold_max_drawdown']:.2%}")


def display_confusion_matrix(predictions, ticker):
    cm    = confusion_matrix(predictions["Target"], predictions["Predictions"])
    cm_df = pd.DataFrame(
        cm,
        index=["Actual ↓", "Actual ↑"],
        columns=["Pred ↓", "Pred ↑"]
    )
    col_cm, col_bar = st.columns([1, 1])

    with col_cm:
        st.markdown(
            "<p style='color:#7fb3cc; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.1em; font-weight:600; margin-bottom:8px;'>CONFUSION MATRIX</p>",
            unsafe_allow_html=True
        )
        st.dataframe(cm_df, use_container_width=True)

    with col_bar:
        counts = predictions["Predictions"].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(4, 2.5))
        bars = ax.bar(
            ["Predicted ↓", "Predicted ↑"],
            [counts.get(0, 0), counts.get(1, 0)],
            color=["#f43f5e", "#00e5a0"],
            width=0.5
        )
        ax.set_title(f"Prediction Distribution — {ticker}", fontsize=9)
        fig, ax = apply_chart_style(fig, ax)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)


def display_feature_importance(importances, ticker):
    top = importances.head(10)

    fig, ax = plt.subplots(figsize=(7, 3))
    colors  = ["#00e5a0" if i == 0 else "#1c3a52" for i in range(len(top))]
    top.sort_values().plot(kind="barh", ax=ax, color=colors[::-1])
    ax.set_title(f"Top 10 Feature Importances — {ticker}", fontsize=9)
    ax.set_xlabel("Importance")
    fig, ax = apply_chart_style(fig, ax)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

    st.caption("Importance from Random Forest — measures average impurity reduction per feature across all trees.")


def display_baseline_comparison(metrics, baseline):
    df = pd.DataFrame({
        "Ensemble": metrics,
        "Always-Up Baseline": baseline
    })
    st.dataframe(df.style.format("{:.3f}"), use_container_width=True)


# ── Main run logic ────────────────────────────────────────────────────────────

if run_button:
    tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]

    if not tickers:
        st.error("Please enter at least one ticker.")
    else:
        summary_rows = []

        for ticker in tickers:
            st.markdown(
                f"<h3 style='font-family: Space Mono, monospace; color:#eaf4ff; font-size:1.2rem; margin-bottom:2px;'>`{ticker}`</h3>",
                unsafe_allow_html=True
            )
            st.markdown("---")

            try:
                threshold = None if threshold_mode == "Adaptive" else manual_threshold

                with st.spinner(f"Running walk-forward backtest for {ticker}…"):
                    result = run_single_ticker(
                        ticker=ticker,
                        threshold=threshold,
                        start=start,
                        step=step
                    )

                metrics          = result["metrics"]
                baseline         = result["baseline_metrics"]
                strategy_results = result["strategy_results"]
                predictions      = result["predictions"]
                importances      = result["importances"]

                # ── Regime + quick metrics ───────────────────────────────────
                display_regime_info(result)
                st.markdown("")
                display_main_metrics(metrics, strategy_results, result)

                st.markdown("---")

                # ── Tabs: Strategy | Models | Diagnostics ───────────────────
                tab_strat, tab_models, tab_diag = st.tabs([
                    "📈  Strategy",
                    "🤖  Model Comparison",
                    "🔍  Diagnostics",
                ])

                with tab_strat:
                    display_strategy_chart(strategy_results)

                with tab_models:
                    display_model_comparison(result)
                    st.markdown("---")
                    st.markdown(
                        "<p style='color:#7fb3cc; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.1em; font-weight:600; margin-bottom:8px;'>VS. ALWAYS-UP BASELINE</p>",
                        unsafe_allow_html=True
                    )
                    display_baseline_comparison(metrics, baseline)

                with tab_diag:
                    col_left, col_right = st.columns(2)
                    with col_left:
                        display_confusion_matrix(predictions, ticker)
                    with col_right:
                        display_feature_importance(importances, ticker)

                # ── Model insight expander ───────────────────────────────────
                with st.expander("ℹ️  Model Architecture Notes"):
                    st.markdown("""
**Random Forest** — strong baseline for noisy tabular data; provides built-in feature importance.

**XGBoost** — gradient-boosted trees; captures non-linear interactions that RF misses.

**Logistic Regression** — linear baseline; useful for checking whether complexity is warranted.

**Ensemble** — probability average of all three; reduces variance and produces more stable signals.

> Walk-forward backtesting retrains each model at every step boundary,
> preventing any lookahead bias. The threshold applied is regime-adaptive
> unless overridden manually.
                    """)

                summary_rows.append({
                    "Ticker":              ticker,
                    "Regime":              result["regime_info"]["regime"],
                    "Threshold":           result["adaptive_settings"]["threshold"],
                    "Accuracy":            metrics["accuracy"],
                    "Precision":           metrics["precision"],
                    "Recall":              metrics["recall"],
                    "F1":                  metrics["f1"],
                    "StrategySharpe":      strategy_results["strategy_sharpe"],
                    "PositionSizeSharpe":  strategy_results["position_size_sharpe"],
                    "BuyHoldSharpe":       strategy_results["buy_hold_sharpe"],
                    "MaxDrawdown":         strategy_results["strategy_max_drawdown"],
                    "WinRate":             strategy_results["win_rate"],
                    "Trades":              strategy_results["number_of_trades"],
                    "LatestSignal":        "UP" if result["latest_signal"] == 1 else "DOWN",
                    "LatestUpProb":        result["latest_prob"],
                })

            except Exception as e:
                st.error(f"**{ticker}** — {e}")

            st.markdown("<br>", unsafe_allow_html=True)

        # ── Cross-ticker summary ─────────────────────────────────────────────
        if len(summary_rows) > 0:
            st.markdown("---")
            st.markdown(
                "<p style='color:#7fb3cc; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.1em; font-weight:600; margin-bottom:10px;'>SUMMARY COMPARISON</p>",
                unsafe_allow_html=True
            )

            summary_df = pd.DataFrame(summary_rows).sort_values(
                by="StrategySharpe", ascending=False
            )

            fmt = {c: "{:.3f}" for c in summary_df.select_dtypes("float").columns}
            fmt["Trades"] = "{:.0f}"

            def highlight_best(s):
                is_max = s == s.max()
                return ["background: rgba(0,229,160,0.25); color: #00e5a0; font-weight: 600;" if v else "" for v in is_max]

            def highlight_worst_drawdown(s):
                is_min = s == s.min()
                return ["background: rgba(251,113,133,0.22); color: #fb7185; font-weight: 600;" if v else "" for v in is_min]

            styled = summary_df.style.format(fmt)
            for col in ["StrategySharpe", "F1", "WinRate"]:
                if col in summary_df.columns:
                    styled = styled.apply(highlight_best, subset=[col])
            if "MaxDrawdown" in summary_df.columns:
                styled = styled.apply(highlight_worst_drawdown, subset=["MaxDrawdown"])

            st.dataframe(styled, use_container_width=True)

            csv = summary_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇  Download CSV",
                data=csv,
                file_name="alphasignal_summary.csv",
                mime="text/csv",
            )

else:
    st.markdown("""
    <div style="margin-top: 80px; text-align: center; user-select: none;">
        <div style="font-size: 3.5rem;">📈</div>
        <div style="color:#a8c4d8; font-size:1rem; margin-top:16px; font-family:'DM Sans',sans-serif; font-weight:500;">
            Configure your tickers and parameters in the sidebar,<br>
            then hit <strong style="color:#00e5a0;">RUN ANALYSIS</strong> to begin.
        </div>
        <div style="color:#64849a; font-size:0.82rem; margin-top:10px;">
            Supports any Yahoo Finance ticker — stocks, ETFs, indices (e.g. ^GSPC)
        </div>
    </div>
    """, unsafe_allow_html=True)