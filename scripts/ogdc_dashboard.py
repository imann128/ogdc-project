"""
ogdc_dashboard.py
=================
Streamlit Dashboard — OGDC Stock Price Analysis

Run with:
    streamlit run ogdc_dashboard.py

Requirements:
    pip install streamlit pandas numpy plotly pillow scipy

Place this file in the scripts/ folder.
"""

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
from scipy import stats as scipy_stats

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="OGDC Stock Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))  # scripts folder
DATA_DIR = os.path.join(BASE, "..", "data", "processed")
IMG_DIR = os.path.join(BASE, "..", "outputs", "images")

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)

# ── Custom CSS — deep teal/electric accent theme ──────────────────────────────
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0a0f1e 0%, #0d1b2a 50%, #091a20 100%);
    min-height: 100vh;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1b2a 0%, #091520 100%);
    border-right: 1px solid rgba(0,212,255,0.13);
}
[data-testid="stHeader"] { background: transparent; }
html, body, [class*="css"] { color: #e8f4f8; font-family: 'Segoe UI', sans-serif; }
h1 { color: #00d4ff !important; font-size: 2.2rem !important; font-weight: 800 !important;
     text-shadow: 0 0 30px #00d4ff55; }
h2 { color: #00d4ff !important; font-size: 1.5rem !important; font-weight: 700 !important; }
h3 { color: #7ee8fa !important; font-size: 1.15rem !important; font-weight: 600 !important; }
p, li, span, label { color: #c8dde8 !important; }
.stMarkdown p { color: #c8dde8 !important; line-height: 1.7; }
[data-testid="stSidebar"] * { color: #a8d8ea !important; }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { color: #00d4ff !important; }
[data-testid="metric-container"] {
    background: linear-gradient(135deg, #0d2233 0%, #0a1a28 100%);
    border: 1px solid rgba(0,212,255,0.2);
    border-radius: 12px;
    padding: 16px 20px !important;
    box-shadow: 0 4px 20px rgba(0,212,255,0.08), inset 0 1px 0 rgba(0,212,255,0.13);
    transition: transform 0.2s, box-shadow 0.2s;
}
[data-testid="metric-container"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 30px #00d4ff25;
}
[data-testid="stMetricLabel"] { color: #7ee8fa !important; font-size: 0.8rem !important;
    font-weight: 600 !important; letter-spacing: 0.05em; text-transform: uppercase; }
[data-testid="stMetricValue"] { color: #00d4ff !important; font-size: 1.8rem !important;
    font-weight: 800 !important; }
[data-testid="stMetricDelta"] { font-size: 0.85rem !important; }
.chart-container {
    background: linear-gradient(135deg, #0d2233 0%, #091820 100%);
    border: 1px solid rgba(0,212,255,0.13);
    border-radius: 14px;
    padding: 16px;
    margin: 10px 0;
    box-shadow: 0 4px 25px #00000055;
    transition: box-shadow 0.2s;
}
.chart-container:hover { box-shadow: 0 6px 35px #00d4ff20; }
.section-header {
    background: linear-gradient(90deg, rgba(0,212,255,0.13) 0%, transparent 100%);
    border-left: 4px solid #00d4ff;
    border-radius: 0 8px 8px 0;
    padding: 10px 18px;
    margin: 20px 0 14px 0;
    font-size: 1.1rem;
    font-weight: 700;
    color: #00d4ff !important;
    letter-spacing: 0.04em;
}
.insight-box {
    background: linear-gradient(135deg, #0d2a1a 0%, #091a14 100%);
    border: 1px solid #00ff8822;
    border-left: 4px solid #00ff88;
    border-radius: 0 10px 10px 0;
    padding: 14px 18px;
    margin: 12px 0;
    color: #a8f0c8 !important;
    font-size: 0.92rem;
    line-height: 1.65;
}
.warning-box {
    background: linear-gradient(135deg, #2a1a0d 0%, #1a1209 100%);
    border-left: 4px solid #ffaa00;
    border-radius: 0 10px 10px 0;
    padding: 14px 18px;
    margin: 12px 0;
    color: #ffe0a0 !important;
    font-size: 0.92rem;
}
[data-testid="stTab"] { background: transparent !important; }
.stTabs [data-baseweb="tab-list"] {
    background: #0d1b2a;
    border-bottom: 2px solid rgba(0,212,255,0.2);
    gap: 4px;
    padding: 0 8px;
}
.stTabs [data-baseweb="tab"] {
    color: #7ee8fa !important;
    font-weight: 600;
    font-size: 0.88rem;
    padding: 10px 18px;
    border-radius: 8px 8px 0 0;
    border: 1px solid transparent;
    transition: all 0.2s;
}
.stTabs [data-baseweb="tab"]:hover { background: rgba(0,212,255,0.08); border-color: rgba(0,212,255,0.13); }
.stTabs [aria-selected="true"] {
    background: linear-gradient(180deg, rgba(0,212,255,0.13) 0%, transparent 100%) !important;
    border-color: #00d4ff44 !important;
    color: #00d4ff !important;
    border-bottom: 2px solid #00d4ff !important;
}
hr { border-color: rgba(0,212,255,0.13) !important; }
[data-testid="stSelectbox"] > div > div {
    background: #0d2233 !important;
    border: 1px solid #00d4ff44 !important;
    border-radius: 8px !important;
    color: #e8f4f8 !important;
}
[data-testid="stExpander"] {
    background: #0d2233 !important;
    border: 1px solid rgba(0,212,255,0.13) !important;
    border-radius: 10px !important;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data
def load_data():
    csv_path = os.path.join(DATA_DIR, "ogdc_cleaned.csv")
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    df.rename(columns={"Return": "Returns", "Change %": "ChangeP", "Vol.": "Volume"}, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    df.dropna(subset=["Price"], inplace=True)
    df["Returns"] = df["Returns"].fillna(0)
    df["Year"] = df["Date"].dt.year
    log_ret = np.log(df["Price"] / df["Price"].shift(1)).fillna(0)
    df["HV20"] = log_ret.rolling(20).std() * np.sqrt(252) * 100
    df["SMA20"] = df["Price"].rolling(20).mean()
    df["SMA50"] = df["Price"].rolling(50).mean()
    df["SMA200"] = df["Price"].rolling(200).mean()
    df["BB_mid"] = df["Price"].rolling(20).mean()
    df["BB_std"] = df["Price"].rolling(20).std()
    df["BB_upper"] = df["BB_mid"] + 2 * df["BB_std"]
    df["BB_lower"] = df["BB_mid"] - 2 * df["BB_std"]
    return df


def load_img(filename):
    path = os.path.join(IMG_DIR, filename)
    if os.path.exists(path):
        return Image.open(path)
    return None


def show_img(filename, caption="", use_container=True):
    img = load_img(filename)
    if img:
        if use_container:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.image(img, caption=caption, use_container_width=True)
        if use_container:
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.warning(f"Chart not found: {filename}")


def insight(text):
    st.markdown(f'<div class="insight-box">💡 {text}</div>', unsafe_allow_html=True)


def warn(text):
    st.markdown(f'<div class="warning-box">⚠️ {text}</div>', unsafe_allow_html=True)


def sec(text):
    st.markdown(f'<div class="section-header">{text}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
df = load_data()

# Pre-compute KPIs
latest_price   = df["Price"].iloc[-1]
start_price    = df["Price"].iloc[0]
total_ret      = (latest_price / start_price - 1) * 100
mean_ret       = df["Returns"].mean()
std_ret        = df["Returns"].std()
sharpe         = mean_ret / std_ret * np.sqrt(252) if std_ret > 0 else 0
max_dd         = ((df["Price"] / df["Price"].cummax()) - 1).min() * 100
current_hv     = df["HV20"].iloc[-1]


# ══════════════════════════════════════════════════════════════════════════════
# MAIN HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("# 📈 OGDC Stock Price Analysis Dashboard")
st.markdown(
    "**Oil & Gas Development Company · Pakistan Stock Exchange · "
    "January 2020 – April 2026**"
)
st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# TABS (simplified - removed ML, Enhanced ML, Sentiment)
# ══════════════════════════════════════════════════════════════════════════════
tabs = st.tabs([
    "🏠 Overview",
    "📊 EDA",
    "📉 Statistics",
    "🌊 GARCH",
    "📈 Trend Analysis",
    "💰 Backtesting",
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
with tabs[0]:
    st.markdown("## 🏠 Project Overview & Key Metrics")

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Latest Price", f"PKR {latest_price:.2f}", f"{df['ChangeP'].iloc[-1]:+.2f}%")
    c2.metric("Total Return", f"{total_ret:+.1f}%", "Jan 2020 → Apr 2026")
    c3.metric("Daily Sharpe", f"{sharpe:.3f}", "Annualised")
    c4.metric("Max Drawdown", f"{max_dd:.1f}%", "Worst peak-to-trough")
    c5.metric("Current HV-20", f"{current_hv:.1f}%", "Annualised volatility")
    c6.metric("Observations", "1,552", "Trading days")

    st.markdown("---")

    sec("📈 Interactive Price Chart")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.7, 0.3], vertical_spacing=0.04)

    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["Price"], name="Close Price",
        line=dict(color="#00d4ff", width=1.5),
        fill="tozeroy", fillcolor="rgba(0,212,255,0.07)"
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["SMA20"], name="SMA-20",
        line=dict(color="#ff8c00", width=1, dash="dot")
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["SMA50"], name="SMA-50",
        line=dict(color="#00ff88", width=1.2, dash="dash")
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["SMA200"], name="SMA-200",
        line=dict(color="#ff4444", width=1.5)
    ), row=1, col=1)

    colors_vol = ["rgba(0,255,136,0.6)" if r >= 0 else "rgba(255,68,68,0.6)"
                  for r in df["Returns"]]
    fig.add_trace(go.Bar(
        x=df["Date"], y=df["Volume"] / 1e6,
        name="Volume (M)", marker_color=colors_vol, showlegend=True
    ), row=2, col=1)

    fig.update_layout(
        paper_bgcolor="#0a0f1e", plot_bgcolor="#0d1b2a",
        font=dict(color="#c8dde8", family="Segoe UI"),
        legend=dict(bgcolor="#0d2233", bordercolor="rgba(0,212,255,0.2)",
                    borderwidth=1, font=dict(size=11)),
        height=520, margin=dict(l=10, r=10, t=10, b=10),
        xaxis2=dict(title="Date", gridcolor="rgba(0,212,255,0.08)",
                    zerolinecolor="rgba(0,212,255,0.13)"),
        yaxis=dict(title="Price (PKR)", gridcolor="rgba(0,212,255,0.08)",
                zerolinecolor="rgba(0,212,255,0.13)"),
        yaxis2=dict(title="Volume (M)", gridcolor="rgba(0,212,255,0.08)"),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    sec("📊 Returns Distribution")
    col1, col2 = st.columns(2)

    with col1:
        fig2 = go.Figure()
        fig2.add_trace(go.Histogram(
            x=df["Returns"].dropna(),
            nbinsx=80, name="Daily Returns",
            marker_color="#00d4ff",
            opacity=0.75,
            marker_line=dict(color="#0a0f1e", width=0.3)
        ))
        mu, sd = df["Returns"].mean(), df["Returns"].std()
        x_r = np.linspace(df["Returns"].quantile(0.001),
                          df["Returns"].quantile(0.999), 300)
        y_norm = scipy_stats.norm.pdf(x_r, mu, sd)
        n_obs = df["Returns"].dropna().shape[0]
        bin_w = (df["Returns"].quantile(0.999) - df["Returns"].quantile(0.001)) / 80
        scale = n_obs * bin_w
        fig2.add_trace(go.Scatter(
            x=x_r, y=y_norm * scale, name="Normal fit",
            line=dict(color="#ff8c00", width=2, dash="dash")
        ))
        fig2.update_layout(
            paper_bgcolor="#0d1b2a", plot_bgcolor="#0d1b2a",
            font=dict(color="#c8dde8"), height=340,
            title=dict(text="Daily Returns Histogram", font=dict(color="#00d4ff")),
            xaxis=dict(title="Return (%)", gridcolor="rgba(0,212,255,0.08)"),
            yaxis=dict(title="Count", gridcolor="rgba(0,212,255,0.08)"),
            legend=dict(bgcolor="#0d2233"),
            margin=dict(l=10, r=10, t=40, b=10)
        )
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        ret_by_year = df.groupby("Year")["Returns"].mean().reset_index()
        fig3 = go.Figure(go.Bar(
            x=ret_by_year["Year"].astype(str),
            y=ret_by_year["Returns"].round(4),
            marker_color=["#ff4444" if v < 0 else "#00d4ff"
                          for v in ret_by_year["Returns"]],
            text=[f"{v:.3f}%" for v in ret_by_year["Returns"]],
            textposition="outside",
            textfont=dict(color="#e8f4f8", size=11),
        ))
        fig3.update_layout(
            paper_bgcolor="#0d1b2a", plot_bgcolor="#0d1b2a",
            font=dict(color="#c8dde8"), height=340,
            title=dict(text="Mean Daily Return by Year", font=dict(color="#00d4ff")),
            xaxis=dict(title="Year", gridcolor="rgba(0,212,255,0.08)"),
            yaxis=dict(title="Mean Return (%)", gridcolor="rgba(0,212,255,0.08)",
                       zeroline=True, zerolinecolor="rgba(0,212,255,0.27)"),
            margin=dict(l=10, r=10, t=40, b=10)
        )
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown("---")
    sec("📋 Key Findings Summary")
    col1, col2 = st.columns(2)
    with col1:
        insight("**Distribution:** OGDC returns follow a **Student-t** distribution "
                "(ν≈2.8 df), NOT Normal. Tail risk is **14× higher** than Gaussian models predict.")
        insight("**ANOVA:** No significant year-on-year mean return difference. High within-year variance dominates.")
    with col2:
        insight("**Volatility Clustering:** ARCH effects strongly confirmed. GJR-GARCH leverage effect γ=0.33 — bad news amplifies volatility more than good news.")
        insight("**Bollinger Bands:** Mean-reversion strategy achieved Sharpe=0.57 with only −16.5% max drawdown vs −41% for Buy & Hold.")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — EDA (STAT Charts)
# ─────────────────────────────────────────────────────────────────────────────
with tabs[1]:
    st.markdown("## 📊 Exploratory Data Analysis")

    sec("EDA Dashboard Panel")
    show_img("stat_01_eda_panel.png",
             "Figure 1: Full EDA panel — price trend, return distribution, "
             "annual boxplots, volume histogram, Q-Q plot, daily return bars")
    insight("The price series shows a clear upward trend from PKR 70 → PKR 334 over 6 years.")

    st.markdown("---")
    sec("Scatter Matrix — Price, Volume, Returns, HV-20")
    show_img("stat_02_scatter_matrix.png",
             "Figure 2: Scatter matrix showing pairwise relationships and Pearson r values")
    insight("Price ~ Volume has a moderate Spearman ρ=0.35 (p<0.001). HV-20 and Returns show near-zero correlation.")

    st.markdown("---")
    sec("Interactive Bollinger Bands — 2024–2026")
    df_zoom = df[df["Date"] >= "2024-01-01"].copy()
    fig_bb = go.Figure()
    fig_bb.add_trace(go.Scatter(
        x=df_zoom["Date"], y=df_zoom["BB_upper"],
        name="Upper Band (+2σ)", line=dict(color="#00d4ff", width=1, dash="dash"),
        showlegend=True
    ))
    fig_bb.add_trace(go.Scatter(
        x=df_zoom["Date"], y=df_zoom["BB_lower"],
        name="Lower Band (−2σ)", line=dict(color="#00d4ff", width=1, dash="dash"),
        fill="tonexty", fillcolor="rgba(0,212,255,0.07)"
    ))
    fig_bb.add_trace(go.Scatter(
        x=df_zoom["Date"], y=df_zoom["BB_mid"],
        name="SMA-20 (Mid)", line=dict(color="#888", width=1)
    ))
    fig_bb.add_trace(go.Scatter(
        x=df_zoom["Date"], y=df_zoom["Price"],
        name="Price", line=dict(color="#00ff88", width=1.8)
    ))
    fig_bb.update_layout(
        paper_bgcolor="#0d1b2a", plot_bgcolor="#0d1b2a",
        font=dict(color="#c8dde8"), height=400,
        xaxis=dict(title="Date", gridcolor="rgba(0,212,255,0.08)"),
        yaxis=dict(title="Price (PKR)", gridcolor="rgba(0,212,255,0.08)"),
        legend=dict(bgcolor="#0d2233"),
        hovermode="x unified",
        margin=dict(l=10, r=10, t=10, b=10)
    )
    st.plotly_chart(fig_bb, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — STATISTICS
# ─────────────────────────────────────────────────────────────────────────────
with tabs[2]:
    st.markdown("## 📉 Advanced Statistical Tests")

    stat_choice = st.selectbox(
        "Select Analysis",
        ["ANOVA (One-Way & Two-Way)",
         "Probability Distribution Fitting",
         "Multiple Regression",
         "ARIMA Time Series",
         "Nonparametric Tests"]
    )

    if stat_choice == "ANOVA (One-Way & Two-Way)":
        sec("ANOVA Results")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("F-Statistic", "0.852")
        col2.metric("p-value", "0.530")
        col3.metric("η² Effect Size", "0.003", "Small")
        col4.metric("Levene's W", "7.38", "Variances unequal")
        show_img("stat_03_anova.png",
                 "Figure 3: One-Way ANOVA by year (left) and Two-Way interaction plot")
        insight("Fail to reject H₀ — no significant year-on-year mean return difference.")
        warn("Levene's test rejects equal variances (p<0.001). Heteroscedasticity across years is real.")

    elif stat_choice == "Probability Distribution Fitting":
        sec("Distribution Fitting — 5 Candidates vs OGDC Returns")
        col1, col2, col3 = st.columns(3)
        col1.metric("Best Fit", "Student-t", "Lowest AIC")
        col2.metric("AIC Gap vs Normal", "−663 pts", "Student-t wins decisively")
        col3.metric("KS p-value (Student-t)", "0.237", "Fail to reject fit ✓")
        show_img("stat_04_distribution_fitting.png",
                 "Figure 4: Five fitted distributions overlaid on return histogram")
        insight("Student-t with ν≈2.8 degrees of freedom means OGDC has extremely fat tails.")

    elif stat_choice == "Multiple Regression":
        sec("OLS Regression: Price ~ Open + High + Low + Volume + Lags")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("R²", "0.9995")
        col2.metric("Adj. R²", "0.9995")
        col3.metric("F-statistic", "481,266")
        col4.metric("RMSE", "1.50 PKR")
        show_img("stat_05_regression.png",
                 "Figure 5: Regression diagnostics — actual vs fitted, residuals, coefficient plot")
        warn("Severe multicollinearity among OHLC predictors (VIF > 1000).")
        insight("Return_lag1 coefficient = −0.053 (p<0.001) — mild negative autocorrelation.")

    elif stat_choice == "ARIMA Time Series":
        sec("ARIMA Modelling — Model Selection & 30-Day Forecast")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Best Model", "ARMA(0,3)")
        col2.metric("AIC", "7,390.80")
        col3.metric("LB Q(20) p-value", "0.703", "White noise ✓")
        col4.metric("30-day Forecast", "0.0748%/day", "Converges to mean")
        show_img("stat_06_arima.png",
                 "Figure 6: ACF/PACF plots, 30-day forecast with 95% CI")
        insight("ARMA(0,3) selected by AIC. Residuals pass all Ljung-Box tests — model is adequate.")

    elif stat_choice == "Nonparametric Tests":
        sec("Nonparametric Statistical Tests")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Kruskal-Wallis H", "3.818", "p=0.701 — NS")
        col2.metric("Mann-Whitney U", "63,578", "p=0.175 — NS")
        col3.metric("Runs Test Z", "0.043", "p=0.966 — Random ✓")
        col4.metric("Spearman ρ (Price~Vol)", "0.351", "p<0.001 ✓")
        show_img("stat_07_nonparametric.png",
                 "Figure 7: KW boxplots, MW violin plot, Spearman heatmap")
        insight("The Runs Test (Z=0.043, p=0.97) directly supports the Efficient Market Hypothesis.")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — GARCH (garch_* charts)
# ─────────────────────────────────────────────────────────────────────────────
with tabs[3]:
    st.markdown("## 🌊 GARCH Volatility Modelling")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Best Model", "GJR-GARCH(1,1)", "Lowest AIC")
    col2.metric("Persistence α+β", "0.78", "Moderate clustering")
    col3.metric("Shock Half-Life", "2.8 days", "Fast mean-reversion")
    col4.metric("Leverage Effect γ", "0.33", "Asymmetric confirmed")

    sec("All Four GARCH Model Volatilities")
    show_img("garch_01_all_models.png",
             "Conditional annualised volatility: ARCH(1), GARCH(1,1), EGARCH(1,1), GJR-GARCH(1,1)")
    insight("GARCH models reveal volatility clusters in 2020 (COVID), mid-2022 (flood crisis), and 2023.")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        sec("Leverage Effect Comparison")
        show_img("garch_02_leverage_comparison.png",
                 "Symmetric (GARCH) vs asymmetric (EGARCH/GJR) conditional volatility")
    with col2:
        sec("30-Day Volatility Forecast")
        show_img("garch_03_forecast.png",
                 "GARCH(1,1) 30-day ahead forecast with regime-coloured bars and 95% confidence band")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        sec("News Impact Curves")
        show_img("garch_04_news_impact.png",
                 "How positive vs negative shocks of equal magnitude affect next-period vol")
        insight("GJR-GARCH news impact: a negative shock of z=−2 generates significantly more volatility than a positive shock of z=+2.")
    with col2:
        sec("Residual Diagnostics")
        show_img("garch_05_diagnostics.png",
                 "Standardised residuals, histogram vs Normal/t, Q-Q plot")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — TREND ANALYSIS (trend_* charts)
# ─────────────────────────────────────────────────────────────────────────────
with tabs[4]:
    st.markdown("## 📈 Trend Analysis — Moving Averages, Bollinger Bands, Volatility")

    trend_choice = st.selectbox(
        "Select Chart Group",
        ["Moving Averages & Cross Signals",
         "Bollinger Bands (Full)",
         "Bollinger Bands (2024–2026 Zoom)",
         "Volatility Panel",
         "Time Series Decomposition",
         "ACF / PACF",
         "Volatility Regimes",
         "Integrated Dashboard"]
    )

    chart_map = {
        "Moving Averages & Cross Signals": ("trend_01_moving_averages.png",
            "SMA 20/50/200, EMA 12/26 with Golden/Death cross markers and volume"),
        "Bollinger Bands (Full)": ("trend_02_bollinger_bands.png",
            "Classic 20-day 2σ Bollinger Bands with %B and bandwidth panels"),
        "Bollinger Bands (2024–2026 Zoom)": ("trend_03_bb_zoom_2024.png",
            "Zoomed view 2024–2026 with mean-reversion buy/sell signals highlighted"),
        "Volatility Panel": ("trend_04_volatility_panel.png",
            "HV-20/60/120, Parkinson, Garman-Klass estimators + volatility regime bands"),
        "Time Series Decomposition": ("trend_05_ts_decomposition.png",
            "Additive decomposition: Observed, Trend (252-day), Seasonal, Residual"),
        "ACF / PACF": ("trend_06_acf_pacf.png",
            "Autocorrelation and partial autocorrelation of OGDC daily returns"),
        "Volatility Regimes": ("trend_07_volatility_regimes.png",
            "Volatility regime classification over time and HV vs |Return| scatter"),
        "Integrated Dashboard": ("trend_08_integrated_dashboard.png",
            "All indicators together: Price + BB + SMA + %B + Bandwidth + HV + Returns"),
    }
    fname, cap = chart_map[trend_choice]
    show_img(fname, cap)

    if trend_choice == "Moving Averages & Cross Signals":
        insight("4 Golden Crosses and 3 Death Crosses detected over the period.")
    elif trend_choice in ["Bollinger Bands (Full)", "Bollinger Bands (2024–2026 Zoom)"]:
        insight("Mean-reversion strategy achieved 56.9% next-day win rate with avg return +0.38%.")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 6 — BACKTESTING (bt_* charts)
# ─────────────────────────────────────────────────────────────────────────────
with tabs[5]:
    st.markdown("## 💰 Bollinger Band Strategy Backtesting")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Transaction Cost", "0.17%", "Per round-trip")
    col2.metric("Stop-Loss", "3%", "Hard stop per trade")
    col3.metric("Train Split", "70%", "Walk-forward OOS")
    col4.metric("Best IS Sharpe", "0.57", "BB Mean Reversion")
    col5.metric("Best IS Max DD", "−16.5%", "vs −41% Buy & Hold")

    sec("Equity Curves — All Strategies (Full Period)")
    show_img("bt_01_equity_curves.png",
             "Full-period equity curves and drawdown panel for all strategies vs benchmarks")
    insight("BB Mean Reversion matches Buy & Hold's Sharpe ratio (0.57) while cutting max drawdown by more than half.")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        sec("Out-of-Sample Performance (2024–2026)")
        show_img("bt_02_oos_curves.png",
                 "OOS equity curves — last 30% of data only")
        insight("BB Breakout showed the best OOS Sharpe (1.04, 66.7% win rate).")
    with col2:
        sec("Strategy Performance Bars")
        show_img("bt_03_performance_bars.png",
                 "Return, Sharpe, Max Drawdown, Win Rate — all strategies vs benchmarks")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        sec("BB Mean-Reversion — Trade Signals (2024–2026)")
        show_img("bt_04_mr_trades.png",
                 "Entry/exit signals plotted on price chart with equity curve below")
    with col2:
        sec("Rolling 12-Month Sharpe")
        show_img("bt_05_rolling_sharpe.png",
                 "Rolling Sharpe ratio for all strategies — shows regime changes over time")


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#3a6a7a; font-size:0.82rem;'>"
    "OGDC Stock Analysis Dashboard · Stock Analysis · 2026 · "
    "Built with Streamlit & Plotly"
    "</p>",
    unsafe_allow_html=True
)