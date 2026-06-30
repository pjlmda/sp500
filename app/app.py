import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pandas.tseries.offsets import BDay

import config

# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="S&P 500 Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

COLORS = {
    "up":       "#00C853",
    "down":     "#FF1744",
    "strategy": "#1976D2",
    "bnh":      "#F57C00",
    "neutral":  "#78909C",
    "grid":     "rgba(200,200,200,0.2)",
}

# ── Cached data loaders ───────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def load_prediction():
    from src.predict import predict_next_day
    return predict_next_day()


@st.cache_data(ttl=3600)
def load_recent_prices(n: int = 120):
    from src.data import load
    return load().tail(n)


@st.cache_data
def load_backtest():
    res_path = os.path.join(config.DATA_DIR, "backtest_results.parquet")
    met_path = os.path.join(config.DATA_DIR, "backtest_metrics.json")
    if not os.path.exists(res_path):
        return None, None
    df = pd.read_parquet(res_path)
    with open(met_path) as f:
        metrics = json.load(f)
    return df, metrics


@st.cache_data
def load_importances():
    reg = joblib.load(os.path.join(config.MODELS_DIR, "regressor.joblib"))
    clf = joblib.load(os.path.join(config.MODELS_DIR, "classifier.joblib"))
    cols = joblib.load(os.path.join(config.MODELS_DIR, "feature_cols.joblib"))
    return pd.DataFrame({
        "feature":    cols,
        "classifier": clf.feature_importances_,
        "regressor":  reg.feature_importances_,
    })


# ── Chart helpers ─────────────────────────────────────────────────────────────

def _apply_layout(fig, title: str = "", height: int = 400):
    fig.update_layout(
        title=title,
        height=height,
        margin=dict(l=10, r=10, t=40, b=10),
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        font=dict(family="Inter, sans-serif", size=13),
        xaxis=dict(showgrid=True, gridcolor=COLORS["grid"], zeroline=False),
        yaxis=dict(showgrid=True, gridcolor=COLORS["grid"], zeroline=False),
    )
    return fig


def chart_price(prices: pd.DataFrame, pred_price: float, pred_date: str):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=prices.index, y=prices["close"],
        mode="lines", name="S&P 500",
        line=dict(color=COLORS["strategy"], width=2),
    ))
    fig.add_trace(go.Scatter(
        x=[pd.Timestamp(pred_date)], y=[pred_price],
        mode="markers", name="Predicted",
        marker=dict(color=COLORS["up"], size=12, symbol="star"),
    ))
    fig.add_vline(
        x=prices.index[-1].timestamp() * 1000,
        line_dash="dash", line_color=COLORS["neutral"], line_width=1,
    )
    _apply_layout(fig, height=380)
    fig.update_yaxes(tickprefix="$", tickformat=",.0f")
    return fig


def chart_cumret(df: pd.DataFrame):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=(df["strategy_cumret"] - 1) * 100,
        mode="lines", name="Strategy (long/flat)",
        line=dict(color=COLORS["strategy"], width=2),
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=(df["bnh_cumret"] - 1) * 100,
        mode="lines", name="Buy & Hold",
        line=dict(color=COLORS["bnh"], width=2, dash="dot"),
    ))
    fig.add_hline(y=0, line_color=COLORS["neutral"], line_width=1)
    _apply_layout(fig, "Cumulative Returns (%)", height=400)
    fig.update_yaxes(ticksuffix="%")
    return fig


def chart_rolling_acc(df: pd.DataFrame):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=df["rolling_accuracy"] * 100,
        mode="lines", name="Rolling 63-day accuracy",
        line=dict(color=COLORS["strategy"], width=2),
        fill="tozeroy",
        fillcolor="rgba(25,118,210,0.1)",
    ))
    fig.add_hline(
        y=50, line_dash="dash",
        line_color=COLORS["neutral"], line_width=1,
        annotation_text="50% baseline",
        annotation_position="bottom right",
    )
    _apply_layout(fig, "Rolling Direction Accuracy (63-day window)", height=300)
    fig.update_yaxes(ticksuffix="%", range=[35, 70])
    return fig


def chart_importance(imp: pd.DataFrame, col: str, title: str):
    df = imp[["feature", col]].sort_values(col)
    fig = go.Figure(go.Bar(
        x=df[col], y=df["feature"],
        orientation="h",
        marker_color=COLORS["strategy"],
        marker_line_width=0,
    ))
    _apply_layout(fig, title, height=max(350, len(df) * 22))
    fig.update_layout(yaxis=dict(tickfont=dict(size=12)))
    return fig


# ── Pages ─────────────────────────────────────────────────────────────────────

def page_prediction():
    pred = load_prediction()
    prices = load_recent_prices(120)

    is_up = pred["predicted_direction"] == "Up"
    dir_color = COLORS["up"] if is_up else COLORS["down"]
    dir_icon  = "▲" if is_up else "▼"
    next_date = (pd.Timestamp(pred["as_of_date"]) + BDay(1)).strftime("%Y-%m-%d")

    st.markdown(f"## Next-Day Prediction &nbsp; <span style='color:{COLORS['neutral']};font-size:0.8em'>as of {pred['as_of_date']} &rarr; {next_date}</span>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Last Close",       f"${pred['last_close']:,.2f}")
    c2.metric("Predicted Price",  f"${pred['predicted_price']:,.2f}",
              delta=f"{pred['predicted_return_pct']:+.2f}%")
    c3.markdown(
        f"""<div style='border:1px solid #e0e0e0; border-radius:8px; padding:12px 16px'>
        <p style='margin:0;font-size:0.85em;color:#666'>Direction</p>
        <p style='margin:0;font-size:2em;font-weight:700;color:{dir_color}'>{dir_icon} {pred['predicted_direction']}</p>
        </div>""",
        unsafe_allow_html=True,
    )
    c4.metric("Confidence", f"{pred['direction_confidence_pct']:.1f}%",
              help="Max probability across Up/Down classes")

    st.markdown("<br>", unsafe_allow_html=True)

    col_chart, col_prob = st.columns([3, 1])
    with col_chart:
        st.plotly_chart(chart_price(prices, pred["predicted_price"], next_date),
                        use_container_width=True)

    with col_prob:
        st.markdown("**Probability breakdown**")
        st.markdown("<br>", unsafe_allow_html=True)
        fig_prob = go.Figure(go.Bar(
            x=[pred["prob_up"], pred["prob_down"]],
            y=["Up", "Down"],
            orientation="h",
            marker_color=[COLORS["up"], COLORS["down"]],
            text=[f"{pred['prob_up']:.1f}%", f"{pred['prob_down']:.1f}%"],
            textposition="outside",
        ))
        fig_prob.update_layout(
            height=200, margin=dict(l=10, r=40, t=10, b=10),
            plot_bgcolor="white", paper_bgcolor="white",
            xaxis=dict(range=[0, 110], showgrid=False, showticklabels=False),
            yaxis=dict(showgrid=False),
            showlegend=False,
        )
        st.plotly_chart(fig_prob, use_container_width=True)


def page_backtest():
    df, metrics = load_backtest()
    if df is None:
        st.warning("No backtest results found. Run `python -m src.backtest` first.")
        return

    st.markdown("## Walk-Forward Backtest Results")
    st.caption(f"Test period: {metrics['test_start']} to {metrics['test_end']}  —  {metrics['n_test_days']} trading days  —  quarterly retraining")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Direction Accuracy", f"{metrics['direction_accuracy']:.2%}")
    c2.metric("Strategy Sharpe",    f"{metrics['strategy_sharpe']:.3f}",
              delta=f"{metrics['strategy_sharpe'] - metrics['bnh_sharpe']:+.3f} vs B&H")
    c3.metric("Strategy Total Ret", f"{metrics['strategy_total_return_pct']:+.1f}%",
              delta=f"{metrics['strategy_total_return_pct'] - metrics['bnh_total_return_pct']:+.1f}% vs B&H")
    c4.metric("Strategy Max DD",    f"{metrics['strategy_max_drawdown_pct']:.1f}%",
              delta=f"{metrics['strategy_max_drawdown_pct'] - metrics['bnh_max_drawdown_pct']:+.1f}% vs B&H",
              delta_color="inverse")

    st.markdown("<br>", unsafe_allow_html=True)
    st.plotly_chart(chart_cumret(df), use_container_width=True)
    st.plotly_chart(chart_rolling_acc(df.dropna(subset=["rolling_accuracy"])),
                    use_container_width=True)

    with st.expander("Full metrics"):
        st.json(metrics)


def page_model():
    imp = load_importances()

    st.markdown("## Model — Feature Importances")
    st.caption(f"{len(imp)} features  |  XGBoost classifier + regressor  |  importance = mean decrease in impurity")

    tab1, tab2 = st.tabs(["Classifier (Direction)", "Regressor (Return)"])
    with tab1:
        st.plotly_chart(
            chart_importance(imp, "classifier", "Classifier feature importances"),
            use_container_width=True,
        )
    with tab2:
        st.plotly_chart(
            chart_importance(imp, "regressor", "Regressor feature importances"),
            use_container_width=True,
        )

    with st.expander("Raw importance table"):
        st.dataframe(
            imp.sort_values("classifier", ascending=False).reset_index(drop=True),
            use_container_width=True,
        )


# ── Sidebar + routing ─────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 📈 S&P 500 Predictor")
    st.markdown("XGBoost model trained on technical indicators + VIX")
    st.markdown("---")
    page = st.radio("", ["Prediction", "Backtest", "Model"], label_visibility="collapsed")
    st.markdown("---")
    if st.button("Refresh data & prediction"):
        st.cache_data.clear()
        st.rerun()

if page == "Prediction":
    page_prediction()
elif page == "Backtest":
    page_backtest()
else:
    page_model()
