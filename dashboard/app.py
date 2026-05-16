"""Streamlit dashboard for Stock Analyzer."""
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# Make `import Stock_Analyzer.*` work no matter where streamlit is launched from.
PACKAGE_ROOT = Path(__file__).resolve().parent.parent  # .../Stock_Analyzer/
REPO_ROOT = PACKAGE_ROOT.parent                        # parent that contains the package
sys.path.insert(0, str(REPO_ROOT))

import Stock_Analyzer.config as config  # noqa: E402
from Stock_Analyzer.src.features import add_technical_indicators  # noqa: E402
from Stock_Analyzer.src.fetcher import fetch_all, fetch_fundamentals, fetch_ticker  # noqa: E402
from Stock_Analyzer.src.predictor import PricePredictor  # noqa: E402

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Stock Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.5rem; padding-bottom: 1rem; }
    div[data-testid="metric-container"] { background: #1e293b; border-radius: 8px; padding: 12px 16px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@st.cache_data(ttl=1800, show_spinner=False)
def cached_ohlcv(ticker: str, period: str) -> pd.DataFrame:
    return fetch_ticker(ticker, period)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_fundamentals(ticker: str) -> dict:
    return fetch_fundamentals(ticker)


@st.cache_resource(show_spinner=False)
def load_predictor(ticker: str) -> PricePredictor | None:
    path = config.MODELS_DIR / f"{ticker}_predictor.pkl"
    if not path.exists():
        return None
    try:
        return PricePredictor.load(ticker)
    except Exception:
        return None


def make_stock_chart(
    df: pd.DataFrame,
    ticker: str,
    indicators: list[str] | None = None,
) -> go.Figure:
    if indicators is None:
        indicators = ["BB", "SMA20", "SMA50"]

    data = add_technical_indicators(df)

    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.52, 0.16, 0.16, 0.16],
        vertical_spacing=0.025,
        subplot_titles=("", "Volume", "RSI (14)", "MACD"),
    )

    # -- Candlestick --
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Price",
            increasing_line_color="#22c55e",
            decreasing_line_color="#ef4444",
            increasing_fillcolor="#22c55e",
            decreasing_fillcolor="#ef4444",
        ),
        row=1, col=1,
    )

    if "BB" in indicators:
        fig.add_trace(
            go.Scatter(
                x=data.index, y=data["bb_upper"], name="BB Upper",
                line=dict(color="rgba(148,163,184,0.5)", width=1, dash="dot"),
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=data.index, y=data["bb_lower"], name="BB Lower",
                line=dict(color="rgba(148,163,184,0.5)", width=1, dash="dot"),
                fill="tonexty",
                fillcolor="rgba(148,163,184,0.08)",
            ),
            row=1, col=1,
        )
    if "SMA20" in indicators:
        fig.add_trace(
            go.Scatter(x=data.index, y=data["sma_20"], name="SMA 20",
                       line=dict(color="#f59e0b", width=1.5)),
            row=1, col=1,
        )
    if "SMA50" in indicators:
        fig.add_trace(
            go.Scatter(x=data.index, y=data["sma_50"], name="SMA 50",
                       line=dict(color="#3b82f6", width=1.5)),
            row=1, col=1,
        )
    if "SMA200" in indicators:
        fig.add_trace(
            go.Scatter(x=data.index, y=data["sma_200"], name="SMA 200",
                       line=dict(color="#a855f7", width=1.5)),
            row=1, col=1,
        )

    # -- Volume --
    bar_colors = [
        "#22c55e" if c >= o else "#ef4444"
        for c, o in zip(df["Close"], df["Open"])
    ]
    fig.add_trace(
        go.Bar(x=df.index, y=df["Volume"], name="Volume", marker_color=bar_colors, opacity=0.55),
        row=2, col=1,
    )

    # -- RSI --
    fig.add_trace(
        go.Scatter(x=data.index, y=data["rsi_14"], name="RSI",
                   line=dict(color="#a78bfa", width=1.5)),
        row=3, col=1,
    )
    for level, color in [(70, "#ef4444"), (30, "#22c55e"), (50, "#475569")]:
        fig.add_hline(y=level, line_dash="dot", line_color=color, line_width=1, row=3, col=1)

    # -- MACD --
    fig.add_trace(
        go.Scatter(x=data.index, y=data["macd"], name="MACD",
                   line=dict(color="#38bdf8", width=1.5)),
        row=4, col=1,
    )
    fig.add_trace(
        go.Scatter(x=data.index, y=data["macd_signal"], name="Signal",
                   line=dict(color="#fb923c", width=1.5)),
        row=4, col=1,
    )
    hist_colors = ["#22c55e" if v >= 0 else "#ef4444" for v in data["macd_hist"]]
    fig.add_trace(
        go.Bar(x=data.index, y=data["macd_hist"], name="Histogram",
               marker_color=hist_colors, opacity=0.6),
        row=4, col=1,
    )

    fig.update_layout(
        template="plotly_dark",
        height=820,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
        margin=dict(t=30, b=10, l=0, r=0),
        xaxis_rangeslider_visible=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def _color_pct(val: object) -> str:
    if isinstance(val, (int, float)):
        return "color: #4ade80" if val > 0 else "color: #f87171" if val < 0 else ""
    return ""


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## 📈 Stock Analyzer")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🌍 Market Overview", "🔍 Stock Analysis", "📊 Prediction Accuracy", "⚙️ Train Models"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    period = st.select_slider(
        "Data period",
        options=["6mo", "1y", "2y", "5y"],
        value=config.DEFAULT_PERIOD,
    )
    if st.button("🔄 Clear cache", use_container_width=True):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.success("Cache cleared")

# ===========================================================================
# PAGE: MARKET OVERVIEW
# ===========================================================================

if page == "🌍 Market Overview":
    st.title("Market Overview")
    st.caption(f"Last 3 months · {len(config.TICKERS)} tickers")

    with st.spinner("Loading market data ..."):
        rows = []
        for ticker in config.TICKERS:
            df_ov = cached_ohlcv(ticker, "3mo")
            if df_ov.empty or len(df_ov) < 10:
                continue
            last = df_ov.iloc[-1]
            prev = df_ov.iloc[-2] if len(df_ov) > 1 else last
            n5 = df_ov.iloc[-6] if len(df_ov) > 5 else df_ov.iloc[0]
            n22 = df_ov.iloc[-22] if len(df_ov) > 22 else df_ov.iloc[0]

            has_model = (config.MODELS_DIR / f"{ticker}_predictor.pkl").exists()

            rows.append(
                {
                    "Ticker": ticker,
                    "Price ($)": round(last["Close"], 2),
                    "1D %": round((last["Close"] - prev["Close"]) / prev["Close"] * 100, 2),
                    "5D %": round((last["Close"] - n5["Close"]) / n5["Close"] * 100, 2),
                    "1M %": round((last["Close"] - n22["Close"]) / n22["Close"] * 100, 2),
                    "Volume": f"{int(last['Volume']):,}",
                    "Predictor": "✅" if has_model else "—",
                }
            )

    ov_df = pd.DataFrame(rows)
    if ov_df.empty:
        st.warning("No data loaded. Check your internet connection.")
        st.stop()

    gainers = (ov_df["1D %"] > 0).sum()
    losers = (ov_df["1D %"] < 0).sum()
    trained = (ov_df["Predictor"] == "✅").sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Stocks", len(ov_df))
    c2.metric("Up today", gainers)
    c3.metric("Down today", losers)
    c4.metric("ML Predictors", f"{trained}/{len(ov_df)}")

    st.markdown("---")

    styled = (
        ov_df.style
        .map(_color_pct, subset=["1D %", "5D %", "1M %"])
        .format({"Price ($)": "${:.2f}", "1D %": "{:+.2f}%", "5D %": "{:+.2f}%", "1M %": "{:+.2f}%"})
    )
    st.dataframe(styled, use_container_width=True, height=900)

# ===========================================================================
# PAGE: STOCK ANALYSIS
# ===========================================================================

elif page == "🔍 Stock Analysis":
    with st.sidebar:
        ticker = st.selectbox("Stock", config.TICKERS)
        indicators = st.multiselect(
            "Overlays",
            ["BB", "SMA20", "SMA50", "SMA200"],
            default=["BB", "SMA20", "SMA50"],
        )

    st.title(f"Stock Analysis — {ticker}")

    with st.spinner(f"Loading {ticker} ..."):
        df = cached_ohlcv(ticker, period)

    if df.empty:
        st.error(
            f"No data for **{ticker}**. "
            "This ticker may be a Fidelity-only mutual fund (FZROX / FBGRX) "
            "not available on Yahoo Finance."
        )
        st.stop()

    fund = cached_fundamentals(ticker)
    prd = load_predictor(ticker)

    # --- Top KPI row ---
    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else last
    chg_1d = (last["Close"] - prev["Close"]) / prev["Close"] * 100
    chg_5d = (last["Close"] - df.iloc[-6]["Close"]) / df.iloc[-6]["Close"] * 100 if len(df) > 5 else 0
    vol_avg = df["Volume"].rolling(20).mean().iloc[-1]
    vol_ratio = last["Volume"] / vol_avg if vol_avg else 1

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Price", f"${last['Close']:.2f}", f"{chg_1d:+.2f}%")
    k2.metric("5-Day", f"{chg_5d:+.2f}%")
    k3.metric("52W High", f"${df['High'].max():.2f}")
    k4.metric("52W Low", f"${df['Low'].min():.2f}")
    k5.metric("P/E", f"{fund['pe_ratio']:.1f}" if fund.get("pe_ratio") else "N/A")
    k6.metric("Beta", f"{fund['beta']:.2f}" if fund.get("beta") else "N/A")

    st.markdown("---")

    # --- Prediction / Fundamentals row ---
    pred_col, fund_col = st.columns([1, 1])

    with pred_col:
        st.subheader("Price Forecast")
        if prd is not None:
            try:
                next_px = prd.predict_next(df)
                chg_pred = (next_px - last["Close"]) / last["Close"] * 100
                st.metric("Next-Day Predicted", f"${next_px:.2f}", f"{chg_pred:+.2f}%")

                hist_pred = prd.predict_history(df).tail(90)
                fig_mini = go.Figure()
                fig_mini.add_trace(
                    go.Scatter(x=hist_pred.index, y=hist_pred["fwd_close"],
                               name="Actual", line=dict(color="#3b82f6"))
                )
                fig_mini.add_trace(
                    go.Scatter(x=hist_pred.index, y=hist_pred["predicted_close"],
                               name="Predicted", line=dict(color="#f59e0b", dash="dot"))
                )
                fig_mini.update_layout(
                    template="plotly_dark", height=220,
                    margin=dict(t=10, b=10, l=0, r=0),
                    showlegend=True,
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig_mini, use_container_width=True)
            except Exception as exc:
                st.warning(f"Prediction error: {exc}")
        else:
            st.info("Train the predictor model for next-day price forecast (see ⚙️ Train Models).")

    with fund_col:
        st.subheader("Fundamentals")
        if fund:
            items = {
                "Name": fund.get("short_name", ticker),
                "Sector": fund.get("sector", "N/A"),
                "Market Cap": (
                    f"${fund['market_cap']/1e9:.1f} B" if fund.get("market_cap") else "N/A"
                ),
                "Fwd P/E": f"{fund['forward_pe']:.1f}" if fund.get("forward_pe") else "N/A",
                "P/B": f"{fund['price_to_book']:.2f}" if fund.get("price_to_book") else "N/A",
                "Div Yield": (
                    f"{fund['dividend_yield']*100:.2f}%"
                    if fund.get("dividend_yield") else "N/A"
                ),
                "Beta": f"{fund['beta']:.2f}" if fund.get("beta") else "N/A",
                "Vol Ratio": f"{vol_ratio:.2f}x (vs 20d avg)",
            }
            for k, v in items.items():
                c_k, c_v = st.columns([1, 2])
                c_k.caption(k)
                c_v.write(v)

    st.markdown("---")

    # --- Main chart ---
    fig_main = make_stock_chart(df, ticker, indicators=indicators)
    st.plotly_chart(fig_main, use_container_width=True)

# ===========================================================================
# PAGE: PREDICTION ACCURACY
# ===========================================================================

elif page == "📊 Prediction Accuracy":
    with st.sidebar:
        ticker = st.selectbox("Stock", config.TICKERS)

    st.title(f"Prediction Accuracy — {ticker}")

    prd = load_predictor(ticker)
    if prd is None:
        st.warning("No trained predictor found. Go to ⚙️ Train Models first.")
        st.stop()

    df = cached_ohlcv(ticker, period)
    if df.empty:
        st.error("No data available.")
        st.stop()

    history = prd.predict_history(df)

    # Error metrics on test portion (last 20%)
    split_idx = int(len(history) * (1 - config.TEST_SIZE))
    test = history.iloc[split_idx:]

    import numpy as np
    mae = (test["predicted_close"] - test["fwd_close"]).abs().mean()
    mape = (((test["predicted_close"] - test["fwd_close"]) / test["fwd_close"]).abs().mean()) * 100
    rmse = float(np.sqrt(((test["predicted_close"] - test["fwd_close"]) ** 2).mean()))
    directional_acc = (
        ((test["predicted_close"].diff() > 0) == (test["fwd_close"].diff() > 0)).mean() * 100
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("MAE", f"${mae:.2f}")
    m2.metric("MAPE", f"{mape:.2f}%")
    m3.metric("RMSE", f"${rmse:.2f}")
    m4.metric("Directional Accuracy", f"{directional_acc:.1f}%")
    st.caption("Metrics computed on hold-out test set (last 20% of data). Directional accuracy = % of days the model correctly predicted up vs down.")

    st.markdown("---")

    # -- Predicted vs Actual chart --
    fig_pred = go.Figure()
    fig_pred.add_trace(
        go.Scatter(x=history.index, y=history["fwd_close"],
                   name="Actual Next-Day Close", line=dict(color="#3b82f6", width=1.5))
    )
    fig_pred.add_trace(
        go.Scatter(x=history.index, y=history["predicted_close"],
                   name="Predicted Next-Day Close", line=dict(color="#f59e0b", dash="dot", width=1.5))
    )
    # Shade the test region
    fig_pred.add_vrect(
        x0=history.index[split_idx], x1=history.index[-1],
        fillcolor="rgba(255,255,255,0.03)", line_width=0,
        annotation_text="Test set", annotation_position="top left",
    )
    fig_pred.update_layout(
        template="plotly_dark",
        title="Predicted vs Actual Next-Day Close",
        height=420,
        margin=dict(t=50, b=10, l=0, r=0),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_pred, use_container_width=True)

    # -- Prediction error over time --
    fig_err = go.Figure()
    err_colors = ["#ef4444" if v > 0 else "#22c55e" for v in history["pred_error_pct"]]
    fig_err.add_trace(
        go.Bar(x=history.index, y=history["pred_error_pct"],
               name="Error %", marker_color=err_colors, opacity=0.7)
    )
    fig_err.update_layout(
        template="plotly_dark",
        title="Prediction Error % (positive = over-predicted)",
        height=280,
        margin=dict(t=50, b=10, l=0, r=0),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_err, use_container_width=True)

# ===========================================================================
# PAGE: TRAIN MODELS
# ===========================================================================

elif page == "⚙️ Train Models":
    st.title("Train Models")
    st.info(
        "Select tickers and a period, then click **Train**. "
        "Downloads price history and trains an XGBoost next-day price predictor per ticker."
    )

    with st.sidebar:
        selected_tickers = st.multiselect(
            "Tickers to train",
            config.TICKERS,
            default=config.TICKERS[:6],
        )
        train_period = st.select_slider("Training period", ["2y", "3y", "5y"], value="5y")
        use_cache_train = st.checkbox("Use cached data", value=True)

    if not selected_tickers:
        st.warning("Select at least one ticker.")
        st.stop()

    if st.button("🚀 Train Selected Models", type="primary", use_container_width=True):
        progress_bar = st.progress(0.0)
        status_box = st.empty()
        results_box = st.empty()
        results: list[dict] = []

        all_data = fetch_all(
            tickers=selected_tickers,
            period=train_period,
            use_cache=use_cache_train,
        )

        for i, ticker in enumerate(selected_tickers):
            progress_bar.progress((i + 1) / len(selected_tickers))
            status_box.info(f"Training **{ticker}** ({i+1}/{len(selected_tickers)}) ...")

            df_t = all_data.get(ticker)
            row: dict = {"Ticker": ticker}

            if df_t is None or len(df_t) < 300:
                row.update({"Status": "❌ No data", "MAPE": "—", "RMSE": "—"})
                results.append(row)
                results_box.dataframe(pd.DataFrame(results), use_container_width=True)
                continue

            try:
                prd_t = PricePredictor(ticker)
                m = prd_t.fit(df_t)
                prd_t.save()
                row.update({
                    "Status": "✅",
                    "MAPE": f"{m['mape']:.2f}%",
                    "RMSE": f"${m['rmse']:.2f}",
                    "Train rows": m["n_train"],
                })
            except Exception as exc:
                row.update({"Status": f"⚠️ {str(exc)[:50]}", "MAPE": "Error", "RMSE": "Error"})

            results.append(row)
            results_box.dataframe(pd.DataFrame(results), use_container_width=True)

        status_box.success("Training complete! Clear the cache (sidebar) so the dashboard reloads fresh models.")

    # Show existing models
    st.markdown("---")
    st.subheader("Existing Models")
    model_rows = [
        {"Ticker": t, "Predictor": "✅"}
        for t in config.TICKERS
        if (config.MODELS_DIR / f"{t}_predictor.pkl").exists()
    ]
    if model_rows:
        st.dataframe(pd.DataFrame(model_rows), use_container_width=True)
    else:
        st.caption("No models trained yet.")
