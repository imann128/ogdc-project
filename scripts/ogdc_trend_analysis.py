"""
ogdc_trend_analysis.py
======================
Financial Market Trend Analysis for OGDC — PSX

Methods Implemented:
  ├── Time Series Analysis
  │     ├── Trend decomposition (additive)
  │     ├── Stationarity (ADF — manual OLS)
  │     └── Autocorrelation / partial autocorrelation (ACF / PACF)
  ├── Moving Averages
  │     ├── Simple Moving Average   — SMA 20, 50, 200
  │     ├── Exponential Moving Average — EMA 12, 26
  │     ├── Weighted Moving Average — WMA 20
  │     └── Golden / Death cross detection
  ├── Bollinger Bands
  │     ├── Classic BB (20-day SMA ± 2σ)
  │     ├── %B indicator (position within bands)
  │     ├── Bandwidth (squeeze detection)
  │     └── Signal generation (breakout / mean-reversion)
  └── Volatility Analysis
        ├── Historical volatility (rolling 20-day annualised)
        ├── Parkinson volatility (High-Low estimator)
        ├── Garman-Klass volatility (OHLC estimator)
        ├── Volatility regimes (low / medium / high)
        └── Volatility clustering (ARCH-effect test)

Output: 8 PNG charts + ogdc_trend_features.csv
"""

import sys, os, shutil, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from paths import processed, img, frontend, PATHS, ensure_dirs
ensure_dirs()

import warnings, os
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import seaborn as sns
from scipy import stats 

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE      = os.path.dirname(os.path.abspath(__file__)) # Scripts
# Data Directory
DATA_DIR = os.path.join(BASE, "..", "data", "processed")  
os.makedirs(DATA_DIR, exist_ok=True)
DATA_PATH = os.path.join(DATA_DIR, "ogdc_cleaned.csv")
# Output Directory
OUTPUT_DIR = os.path.join(BASE, "..", "data", "processed")
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUT_CSV = os.path.join(OUTPUT_DIR, "ogdc_trend_features.csv")
# Image Directory
IMG_DIR = os.path.join(BASE, "..", "outputs", "images")
os.makedirs(IMG_DIR, exist_ok=True)
IMG_PRE = os.path.join(IMG_DIR, "trend_")


plt.rcParams.update({
    "figure.facecolor": "#0d1117",
    "axes.facecolor":   "#161b22",
    "axes.edgecolor":   "#30363d",
    "axes.labelcolor":  "#e6edf3",
    "xtick.color":      "#8b949e",
    "ytick.color":      "#8b949e",
    "text.color":       "#e6edf3",
    "grid.color":       "#21262d",
    "grid.linewidth":   0.6,
    "legend.facecolor": "#161b22",
    "legend.edgecolor": "#30363d",
})

C = {   # colour palette
    "price":    "#58a6ff",
    "sma20":    "#f0883e",
    "sma50":    "#3fb950",
    "sma200":   "#ff7b72",
    "ema12":    "#d2a8ff",
    "ema26":    "#ffa657",
    "wma20":    "#79c0ff",
    "bb_upper": "#388bfd",
    "bb_lower": "#388bfd",
    "bb_fill":  "#1f4080",
    "vol_hist": "#2ea043",
    "vol_park": "#d29922",
    "vol_gk":   "#a371f7",
    "regime_lo":"#2ea043",
    "regime_md":"#d29922",
    "regime_hi":"#f85149",
    "signal_buy":"#2ea043",
    "signal_sell":"#f85149",
    "squeeze":  "#ffa657",
    "text":     "#e6edf3",
    "grid":     "#21262d",
}

# Saves images to image directory
def save_fig(fig, name):
    path = IMG_PRE + name
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f" saved: trend_{name}")

# Adds title to images
def banner(title):
    bar = "═" * 70
    print(f"\n{bar}\n  {title}\n{bar}")
# Adds subtitle to images
def sub(title):
    print(f"\n  {'─'*58}\n  {title}\n  {'─'*58}")

# ══════════════════════════════════════════════════════════════════════════════
# Load data
# ══════════════════════════════════════════════════════════════════════════════
banner("Loading Data")
# Reads data from the csv
df = pd.read_csv(DATA_PATH, parse_dates=["Date"], index_col="Date")
df.sort_index(inplace=True)
df.rename(columns={"Return": "Returns", "Change %": "ChangeP", "Vol.": "Volume"}, inplace=True)
df.dropna(subset=["Price"], inplace=True)

# Fill missing OHLC with Price for robustness
for c in ["Open","High","Low"]:
    if c not in df.columns:
        df[c] = df["Price"]
    df[c] = df[c].fillna(df["Price"])
# Hadnling empty vales
df["Returns"] = df["Returns"].fillna(0)
print(f"  Rows: {len(df)}   Range: {df.index.min().date()} → {df.index.max().date()}")
print(f"  Columns: {list(df.columns)}")

# Working series
price  = df["Price"]
high   = df["High"]
low    = df["Low"]
open_  = df["Open"]
volume = df["Volume"].fillna(0)
ret    = df["Returns"]          # daily % return

# ══════════════════════════════════════════════════════════════════════════════
# Part A — Moving Averages
# ══════════════════════════════════════════════════════════════════════════════
banner("Part A — Moving Averages")

# ── Simple Moving Averages ────────────────────────────────────────────────────
df["SMA_20"]  = price.rolling(20).mean()
df["SMA_50"]  = price.rolling(50).mean()
df["SMA_200"] = price.rolling(200).mean()

# ── Exponential Moving Averages ───────────────────────────────────────────────
df["EMA_12"] = price.ewm(span=12, adjust=False).mean()
df["EMA_26"] = price.ewm(span=26, adjust=False).mean()

# ── Weighted Moving Average (linearly weighted) ────────────────────────────────
def wma(series, window):
    weights = np.arange(1, window + 1, dtype=float)
    return series.rolling(window).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

df["WMA_20"] = wma(price, 20)

# ── Golden Cross / Death Cross detection ──────────────────────────────────────
# Golden Cross: SMA_50 crosses above SMA_200 → bullish
# Death Cross:  SMA_50 crosses below SMA_200 → bearish
sma50  = df["SMA_50"].dropna()
sma200 = df["SMA_200"].dropna()
aligned = sma50.index.intersection(sma200.index)
s50 = sma50.loc[aligned]; s200 = sma200.loc[aligned]

cross_signal = np.sign(s50 - s200)
cross_change = cross_signal.diff() # Buy order and sell order at the same security
golden_crosses = aligned[cross_change > 0] # Potential shift to a long-term uptrend
death_crosses  = aligned[cross_change < 0] # Potential bearish trend

print(f"  SMA 20/50/200, EMA 12/26, WMA 20 computed")
print(f"  Golden Crosses detected: {len(golden_crosses)}")
print(f"  Death  Crosses detected: {len(death_crosses)}")

# ── Price-MA relationship stats ────────────────────────────────────────────────
sub("Price vs SMA position")
for col, label in [("SMA_20","SMA-20"),("SMA_50","SMA-50"),("SMA_200","SMA-200")]:
    above = (price > df[col]).sum()
    total = df[col].notna().sum()
    pct   = above / total * 100
    print(f"  Price above {label}: {pct:.1f}% of days ({above}/{total})")


# ══════════════════════════════════════════════════════════════════════════════
# Part B — Bollinger Bands
# ══════════════════════════════════════════════════════════════════════════════
banner("Part B — Bollinger Bands")

# Classic Bollinger Bands: middle = SMA-20, upper/lower = middle ± 2σ
BB_WINDOW = 20
BB_STD    = 2

df["BB_mid"]   = price.rolling(BB_WINDOW).mean()
bb_std         = price.rolling(BB_WINDOW).std()
df["BB_upper"] = df["BB_mid"] + BB_STD * bb_std
df["BB_lower"] = df["BB_mid"] - BB_STD * bb_std

# %B indicator: position of price within the band (0 = lower, 1 = upper)
df["BB_pct_b"] = (price - df["BB_lower"]) / (df["BB_upper"] - df["BB_lower"])

# Bandwidth: (upper - lower) / middle  — squeeze = low bandwidth
df["BB_bandwidth"] = (df["BB_upper"] - df["BB_lower"]) / df["BB_mid"]

# Squeeze detection: bandwidth < 20th percentile of its own history
bw_20th = df["BB_bandwidth"].rolling(120).quantile(0.20)
df["BB_squeeze"] = df["BB_bandwidth"] < bw_20th

# Signal generation
# Breakout long:  close > upper band (momentum / breakout strategy)
# Breakout short: close < lower band
# Mean-revert buy:  %B < 0.05 (price near / below lower band)
# Mean-revert sell: %B > 0.95 (price near / above upper band)
df["BB_sig_breakout_long"]  = (price > df["BB_upper"]).astype(int)
df["BB_sig_breakout_short"] = (price < df["BB_lower"]).astype(int)
df["BB_sig_mr_buy"]         = (df["BB_pct_b"] < 0.05).astype(int)
df["BB_sig_mr_sell"]        = (df["BB_pct_b"] > 0.95).astype(int)

# Stats
squeeze_days = df["BB_squeeze"].sum() # Rapid increase in the stock price due to excess of short selling
bo_long_days = df["BB_sig_breakout_long"].sum() # Stock price suddenly shoots up or down with massive force
bo_shrt_days = df["BB_sig_breakout_short"].sum() # Sudden upward move after a period of selling, overcomes resistance
mr_buy_days  = df["BB_sig_mr_buy"].sum() # Purchasing a stock after it's price has declined significantly
mr_sell_days = df["BB_sig_mr_sell"].sum() # Selling a stock after it's price has increased too much

print(f"  Bollinger Bands (20-day, 2σ) computed")
print(f"  Squeeze days (BW < 20th pct)  : {squeeze_days}")
print(f"  Breakout-long days (>upper)    : {bo_long_days}")
print(f"  Breakout-short days (<lower)   : {bo_shrt_days}")
print(f"  Mean-revert buy signals        : {mr_buy_days}")
print(f"  Mean-revert sell signals       : {mr_sell_days}")

sub("Bollinger Band Signal Effectiveness")
# Check: did price go up the day after a BB signal?
for sig, label in [
    ("BB_sig_breakout_long","Breakout Long"),
    ("BB_sig_breakout_short","Breakout Short"),
    ("BB_sig_mr_buy","MR Buy"),
    ("BB_sig_mr_sell","MR Sell"),
]:
    mask = df[sig].shift(1) == 1   # signal was active yesterday
    n = mask.sum()
    if n > 0:
        next_ret = ret[mask]
        win_pct  = (next_ret > 0).mean() * 100
        avg_ret  = next_ret.mean()
        print(f"    {label:25s}: n={n:4d}  next-day win%={win_pct:.1f}%  avg_ret={avg_ret:+.3f}%")


# ══════════════════════════════════════════════════════════════════════════════
# Part C — Volatility Analysis
# ══════════════════════════════════════════════════════════════════════════════
banner("Part C — Volatility Analysis")

TRADING_DAYS = 252   # annualisation factor

# ── Historical Volatility (close-to-close, rolling 20-day, annualised) ─────────
log_ret = np.log(price / price.shift(1))
df["HV_20"]   = log_ret.rolling(20).std()  * np.sqrt(TRADING_DAYS) * 100
df["HV_60"]   = log_ret.rolling(60).std()  * np.sqrt(TRADING_DAYS) * 100
df["HV_120"]  = log_ret.rolling(120).std() * np.sqrt(TRADING_DAYS) * 100

# ── Parkinson Volatility (uses High-Low range — more efficient than close-close)
''' Method for estimating an asset's price volatility using it's daily low and high prices 
rather than just the closing prices. Used to capture intraday swings and easier to apply on high and low prices.
'''
# σ_P = sqrt( 1/(4n·ln2) · Σ ln(H/L)² )
hl_log = np.log(high / low)
df["HV_Parkinson"] = np.sqrt(
    hl_log.pow(2).rolling(20).mean() / (4 * np.log(2))
) * np.sqrt(TRADING_DAYS) * 100

# ── Garman-Klass Volatility (OHLC — most efficient classic estimator) ──────────
'''
Method to estimate an asset's historical volatility using high, low, open, and closing prices.
Valuable for risk management and helps get price dispersion within a trading period compared to 
close-to-close methods.
'''
# σ_GK² = 0.5·[ln(H/L)]² − (2·ln2−1)·[ln(C/O)]²
hl2  = 0.5 * np.log(high / low).pow(2)
co2  = (2 * np.log(2) - 1) * np.log(price / open_).pow(2)
df["HV_GarmanKlass"] = np.sqrt(
    (hl2 - co2).rolling(20).mean()
) * np.sqrt(TRADING_DAYS) * 100

# ── Volatility Regimes ─────────────────────────────────────────────────────────
hv = df["HV_20"].dropna()
q33 = hv.quantile(0.33)
q67 = hv.quantile(0.67)

df["Vol_regime"] = "Medium"
df.loc[df["HV_20"] <= q33, "Vol_regime"] = "Low"
df.loc[df["HV_20"] >  q67, "Vol_regime"] = "High"

print(f"  Historical Volatility (20/60/120-day annualised) computed")
print(f"  Parkinson Volatility computed")
print(f"  Garman-Klass Volatility computed")
print(f"\n  Volatility Regime thresholds:")
print(f"    Low    : HV_20 ≤ {q33:.1f}%")
print(f"    Medium : {q33:.1f}% < HV_20 ≤ {q67:.1f}%")
print(f"    High   : HV_20 > {q67:.1f}%")

for r in ["Low","Medium","High"]:
    n = (df["Vol_regime"] == r).sum()
    print(f"    {r:8s}: {n} days ({100*n/len(df):.1f}%)")

# ── ARCH Effect Test (volatility clustering) ────────────────────────────────────
'''
Autoregressive Conditional Heteroskedasticity Effect test is used to find an effect 
in time series data where volatility is not constant over time, but instead clusters together
'''
sub("ARCH Effect Test (Volatility Clustering)")
# Ljung-Box on squared returns — if significant → ARCH effects present
'''
Ljung Box is a type of statistical test of whether any of group of autocorrelations of a time series
are different from zero. It uses lags and determines whether or not the errors are white noise
'''
sq_ret = ret.dropna() ** 2
n = len(sq_ret)
acf_vals = []
for k in range(1, 11):
    r_k = np.corrcoef(sq_ret[k:], sq_ret[:-k])[0, 1]
    acf_vals.append(r_k)
# Formula where r is the accumulated sample of autocorrelations and n - k is time lag
Q10 = n * (n + 2) * sum(r**2 / (n - k) for k, r in enumerate(acf_vals, 1))
p_arch = 1 - stats.chi2.cdf(Q10, df=10)
print(f"  Ljung-Box Q(10) on squared returns: {Q10:.2f}")
print(f"  p-value: {p_arch:.6f}")
if p_arch < 0.05:
    print("   ARCH effects present — volatility clusters in time (expected for equity data).")
else:
    print("  No significant ARCH effects detected.")

# ── Volatility vs Returns correlation ─────────────────────────────────────────
sub("Volatility-Return Relationship")
for hv_col, label in [("HV_20","HV-20"),("HV_Parkinson","Parkinson"),("HV_GarmanKlass","Garman-Klass")]:
    corr = df[hv_col].corr(df["Returns"])
    print(f"  Corr({label}, Returns): {corr:+.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# Part D — Time Series Analysis
# ══════════════════════════════════════════════════════════════════════════════
banner("Part D — Time Series Analysis")

# ── Trend decomposition ─────────────────────────────────────────────────
# Additive model: Y = Trend + Seasonal + Residual
# Trend:    centred 252-day rolling mean (1 trading year)
# Seasonal: Y - Trend  then averaged by day-of-year
# Residual: Y - Trend - Seasonal

# decompose price into trend, seasonal, and residual components
sub("Trend Decomposition (Additive, 252-day window)")
trend    = price.rolling(252, center=True, min_periods=126).mean()
detrended = price - trend

# Seasonal component: group detrended by day-of-year, take mean
doy = detrended.index.dayofyear
seasonal_mean = detrended.groupby(doy).transform("mean")
residual = price - trend - seasonal_mean

df["Trend"]    = trend
df["Seasonal"] = seasonal_mean
df["Residual"] = residual

print(f"  Trend (252-day centred MA) computed")
print(f"  Seasonal component: std = {seasonal_mean.std():.4f}")
print(f"  Residual: std = {residual.std():.4f}, mean = {residual.mean():.4f}")

# ──  ADF stationarity test ──────────────────────────────────────────────
sub("ADF Stationarity Tests")
'''
Checks if time series are stationary i.e. mean, variance, and autocorrelation constant over time
'''
def adf_pvalue(series, max_lag=10):
    y = series.dropna().values.astype(float)
    n = len(y)
    best_lag, best_aic = 1, np.inf
    for lag in range(1, max_lag + 1):
        dy = np.diff(y)
        rows = []
        for i in range(lag, len(dy)):
            r = [y[i]] + [dy[i-k] for k in range(1, lag+1)] + [1]
            rows.append(r)
        X = np.array(rows); yy = dy[lag:]
        coef, *_ = np.linalg.lstsq(X, yy, rcond=None)
        resid = yy - X @ coef
        rss = float(resid @ resid)
        aic = len(yy) * np.log(max(rss/len(yy), 1e-12)) + 2*X.shape[1]
        if aic < best_aic:
            best_aic, best_lag = aic, lag
    lag = best_lag
    dy = np.diff(y)
    rows = []
    for i in range(lag, len(dy)):
        r = [y[i]] + [dy[i-k] for k in range(1, lag+1)] + [1]
        rows.append(r)
    X = np.array(rows); yy = dy[lag:]
    coef, *_ = np.linalg.lstsq(X, yy, rcond=None)
    resid = yy - X @ coef
    rss = float(resid @ resid)
    s2 = rss / max(len(yy) - X.shape[1], 1)
    try:
        cov = s2 * np.linalg.inv(X.T @ X)
        se = np.sqrt(cov[0, 0])
    except Exception:
        se = np.nan
    stat = coef[0] / se if (se and not np.isnan(se)) else np.nan
    pval = 2 * stats.t.cdf(stat, df=max(len(yy)-X.shape[1], 1))
    return stat, pval

for series, label in [
    (price,              "Price (level)"),
    (ret,                "Returns (daily %)"),
    (log_ret,            "Log Returns"),
    (df["HV_20"],        "HV-20 (volatility)"),
]:
    stat, pval = adf_pvalue(series)
    result = "It is stationary" if pval < 0.05 else "It is not stationary"
    print(f"  {label:30s}: ADF={stat:8.3f}  p={pval:.4f}  → {result}")

# ── ACF / PACF ─────────────────────────────────────────────────────────
sub("ACF / PACF of Returns (lags 1-30)")
# compute autocorrelation and partial autocorrelation up to 30 lags
def compute_acf(series, max_lag=30):
    x = series.dropna().values
    n = len(x)
    xm = x - x.mean()
    c0 = xm @ xm / n
    acf = [1.0]
    for k in range(1, max_lag + 1):
        c_k = (xm[k:] @ xm[:-k]) / n
        acf.append(c_k / c0)
    return np.array(acf)

def compute_pacf(series, max_lag=30):
    # Yule-Walker PACF via recursive Levinson-Durbin.
    '''
    calculates partial autocorrelation (correlation between a time series and its lagged values, 
    removing the effect of all shorter lags) using an efficient step-by-step algorithm
    '''
    acf = compute_acf(series, max_lag)
    pacf = [1.0]
    phi = {}
    for k in range(1, max_lag + 1):
        if k == 1:
            p = acf[1]
        else:
            num = acf[k] - sum(phi[k-1][j] * acf[k-1-j] for j in range(k-1))
            den = 1 - sum(phi[k-1][j] * acf[j+1] for j in range(k-1))
            p = num / den if abs(den) > 1e-12 else 0
        phi[k] = [p]
        for j in range(k-1):
            phi[k].append(phi[k-1][j] - p * phi[k-1][k-2-j])
        pacf.append(p)
    return np.array(pacf)

# calculate acf/pacf and identify significant lags at 95% confidence
acf_vals  = compute_acf(ret, 30)
pacf_vals = compute_pacf(ret, 30)
n_ret     = ret.dropna().shape[0]
conf95    = 1.96 / np.sqrt(n_ret)

sig_acf  = np.where(np.abs(acf_vals[1:]) > conf95)[0] + 1
sig_pacf = np.where(np.abs(pacf_vals[1:]) > conf95)[0] + 1
print(f"  95% confidence bound: ±{conf95:.4f}")
print(f"  Significant ACF  lags: {sig_acf.tolist()}")
print(f"  Significant PACF lags: {sig_pacf.tolist()}")


# ══════════════════════════════════════════════════════════════════════════════
# Feature table
# ══════════════════════════════════════════════════════════════════════════════
banner("Saving Feature Table")

keep_cols = [
    "Price","Open","High","Low","Volume","Returns",
    "SMA_20","SMA_50","SMA_200","EMA_12","EMA_26","WMA_20",
    "BB_mid","BB_upper","BB_lower","BB_pct_b","BB_bandwidth","BB_squeeze",
    "BB_sig_breakout_long","BB_sig_breakout_short","BB_sig_mr_buy","BB_sig_mr_sell",
    "HV_20","HV_60","HV_120","HV_Parkinson","HV_GarmanKlass","Vol_regime",
    "Trend","Seasonal","Residual",
]
feature_df = df[[c for c in keep_cols if c in df.columns]].copy()
feature_df.to_csv(OUT_CSV)
print(f"  Saved {len(feature_df)} rows × {len(feature_df.columns)} columns → ogdc_trend_features.csv")


# ══════════════════════════════════════════════════════════════════════════════
# Visualisations  
# ══════════════════════════════════════════════════════════════════════════════
banner("Generating Charts")

# ── Chart 1: Moving Averages Overview ─────────────────────────────────────────
fig = plt.figure(figsize=(16, 9))
gs  = gridspec.GridSpec(3, 1, height_ratios=[4, 1.2, 1.2], hspace=0.08)

ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1], sharex=ax1)
ax3 = fig.add_subplot(gs[2], sharex=ax1)

# Price + MAs
ax1.plot(price.index, price,           lw=1.0, color=C["price"],  alpha=0.9, label="Price", zorder=3)
ax1.plot(df.index, df["SMA_20"],       lw=1.2, color=C["sma20"],  alpha=0.85, label="SMA-20")
ax1.plot(df.index, df["SMA_50"],       lw=1.4, color=C["sma50"],  alpha=0.85, label="SMA-50")
ax1.plot(df.index, df["SMA_200"],      lw=1.8, color=C["sma200"], alpha=0.85, label="SMA-200")
ax1.plot(df.index, df["EMA_12"],       lw=0.9, color=C["ema12"],  alpha=0.7,  linestyle="--", label="EMA-12")
ax1.plot(df.index, df["EMA_26"],       lw=0.9, color=C["ema26"],  alpha=0.7,  linestyle="--", label="EMA-26")

# Golden / Death crosses
for dt in golden_crosses:
    ax1.axvline(dt, color=C["signal_buy"],  alpha=0.35, lw=1.0, linestyle=":")
for dt in death_crosses:
    ax1.axvline(dt, color=C["signal_sell"], alpha=0.35, lw=1.0, linestyle=":")

gc_patch = mpatches.Patch(color=C["signal_buy"],  alpha=0.6, label=f"Golden Cross ({len(golden_crosses)})")
dc_patch = mpatches.Patch(color=C["signal_sell"], alpha=0.6, label=f"Death Cross ({len(death_crosses)})")
ax1.legend(handles=ax1.get_legend_handles_labels()[0] + [gc_patch, dc_patch],
           fontsize=8, loc="upper left", ncol=4)
ax1.set_ylabel("Price (PKR)", fontsize=10)
ax1.set_title("OGDC — Moving Averages & Cross Signals  (2020–2026)", fontsize=13, fontweight="bold", pad=10)
ax1.grid(True, alpha=0.3)
plt.setp(ax1.get_xticklabels(), visible=False)

# SMA spread (50 - 200)
spread = df["SMA_50"] - df["SMA_200"]
ax2.plot(spread.index, spread, lw=1.0, color="#79c0ff")
ax2.fill_between(spread.index, spread, 0,
                 where=spread > 0, color=C["signal_buy"],  alpha=0.25, label="Bullish (SMA50>200)")
ax2.fill_between(spread.index, spread, 0,
                 where=spread < 0, color=C["signal_sell"], alpha=0.25, label="Bearish (SMA50<200)")
ax2.axhline(0, color="#8b949e", lw=0.8)
ax2.set_ylabel("SMA50−200", fontsize=9)
ax2.legend(fontsize=8, loc="upper left")
ax2.grid(True, alpha=0.3)
plt.setp(ax2.get_xticklabels(), visible=False)

# Volume
vol_colors = [C["signal_buy"] if r >= 0 else C["signal_sell"] for r in ret.reindex(df.index).fillna(0)]
ax3.bar(df.index, volume / 1e6, width=1.5, color=vol_colors, alpha=0.7)
ax3.set_ylabel("Volume (M)", fontsize=9)
ax3.set_xlabel("Date", fontsize=9)
ax3.grid(True, alpha=0.3)

plt.setp(ax1.get_xticklabels(), visible=False)
plt.setp(ax2.get_xticklabels(), visible=False)
save_fig(fig, "01_moving_averages.png")

# ── Chart 2: Bollinger Bands ───────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(16, 10),
                          gridspec_kw={"height_ratios":[4,1.2,1.2]}, sharex=True)
fig.subplots_adjust(hspace=0.06)

ax1, ax2, ax3 = axes

# Main BB chart
ax1.plot(price.index, price,          lw=1.0, color=C["price"],    zorder=3, label="Price")
ax1.plot(df.index, df["BB_upper"],    lw=1.2, color=C["bb_upper"], linestyle="--", alpha=0.85, label="Upper Band (+2σ)")
ax1.plot(df.index, df["BB_mid"],      lw=1.0, color="#8b949e",     linestyle="-",  alpha=0.7,  label="Middle (SMA-20)")
ax1.plot(df.index, df["BB_lower"],    lw=1.2, color=C["bb_lower"], linestyle="--", alpha=0.85, label="Lower Band (−2σ)")
ax1.fill_between(df.index, df["BB_upper"], df["BB_lower"],
                 color=C["bb_fill"], alpha=0.15)

# Squeeze shading
sq_mask = df["BB_squeeze"].fillna(False)
ax1.fill_between(df.index, df["BB_lower"], df["BB_upper"],
                 where=sq_mask, color=C["squeeze"], alpha=0.18, label="Squeeze Zone")

# Signal markers
bo_long_idx  = df.index[df["BB_sig_breakout_long"]  == 1]
bo_short_idx = df.index[df["BB_sig_breakout_short"] == 1]
mr_buy_idx   = df.index[df["BB_sig_mr_buy"]         == 1]
mr_sell_idx  = df.index[df["BB_sig_mr_sell"]         == 1]

ax1.scatter(bo_long_idx,  price.reindex(bo_long_idx),  marker="^", color=C["signal_buy"],  s=30, zorder=5, alpha=0.7, label=f"Breakout Long ({len(bo_long_idx)})")
ax1.scatter(bo_short_idx, price.reindex(bo_short_idx), marker="v", color=C["signal_sell"], s=30, zorder=5, alpha=0.7, label=f"Breakout Short ({len(bo_short_idx)})")

ax1.set_title("OGDC — Bollinger Bands (20-day, 2σ)  |  Breakout & Mean-Reversion Signals", fontsize=13, fontweight="bold", pad=10)
ax1.set_ylabel("Price (PKR)", fontsize=10)
ax1.legend(fontsize=7.5, loc="upper left", ncol=4)
ax1.grid(True, alpha=0.3)

# %B
ax2.plot(df.index, df["BB_pct_b"], lw=0.9, color="#79c0ff")
ax2.fill_between(df.index, df["BB_pct_b"], 0.5,
                 where=df["BB_pct_b"] > 0.5, color=C["signal_buy"],  alpha=0.2)
ax2.fill_between(df.index, df["BB_pct_b"], 0.5,
                 where=df["BB_pct_b"] < 0.5, color=C["signal_sell"], alpha=0.2)
ax2.axhline(0,    color="#f85149", lw=0.8, linestyle="--", alpha=0.7)
ax2.axhline(0.5,  color="#8b949e", lw=0.8)
ax2.axhline(1,    color="#2ea043", lw=0.8, linestyle="--", alpha=0.7)
ax2.set_ylabel("%B", fontsize=9)
ax2.set_ylim(-0.3, 1.3)
ax2.grid(True, alpha=0.3)

# Bandwidth
ax3.plot(df.index, df["BB_bandwidth"] * 100, lw=0.9, color=C["squeeze"], label="Bandwidth (%)")
ax3.fill_between(df.index, df["BB_bandwidth"] * 100, 0, alpha=0.2, color=C["squeeze"])
ax3.set_ylabel("Bandwidth %", fontsize=9)
ax3.set_xlabel("Date", fontsize=9)
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)

save_fig(fig, "02_bollinger_bands.png")

# ── Chart 3: Bollinger Band signals — zoomed (2024–2026) ────────────────────
fig, ax = plt.subplots(figsize=(15, 6))
zoom = df["2024":]
zp   = price["2024":]

ax.plot(zp.index, zp,                        lw=1.2, color=C["price"],    zorder=3, label="Price")
ax.plot(zoom.index, zoom["BB_upper"],         lw=1.2, color=C["bb_upper"], linestyle="--", alpha=0.85, label="Upper Band")
ax.plot(zoom.index, zoom["BB_mid"],           lw=0.9, color="#8b949e",     alpha=0.7,      label="SMA-20")
ax.plot(zoom.index, zoom["BB_lower"],         lw=1.2, color=C["bb_lower"], linestyle="--", alpha=0.85, label="Lower Band")
ax.fill_between(zoom.index, zoom["BB_upper"], zoom["BB_lower"], color=C["bb_fill"], alpha=0.2)

# MR signals in zoom
mr_b = zoom.index[zoom["BB_sig_mr_buy"]  == 1]
mr_s = zoom.index[zoom["BB_sig_mr_sell"] == 1]
ax.scatter(mr_b, zp.reindex(mr_b), marker="o", color=C["signal_buy"],  s=60, zorder=5, label=f"MR Buy ({len(mr_b)})")
ax.scatter(mr_s, zp.reindex(mr_s), marker="o", color=C["signal_sell"], s=60, zorder=5, label=f"MR Sell ({len(mr_s)})")

# Squeeze zones in zoom
sq_z = zoom["BB_squeeze"].fillna(False)
ax.fill_between(zoom.index, zoom["BB_lower"], zoom["BB_upper"],
                where=sq_z, color=C["squeeze"], alpha=0.2, label="Squeeze")

ax.set_title("Bollinger Bands — 2024–2026 Zoom  |  Mean-Reversion Signals", fontsize=13, fontweight="bold")
ax.set_ylabel("Price (PKR)")
ax.legend(fontsize=9, ncol=4)
ax.grid(True, alpha=0.3)
save_fig(fig, "03_bb_zoom_2024.png")

# ── Chart 4: Volatility Panel ─────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(16, 11),
                          gridspec_kw={"height_ratios":[2,2,1.5]}, sharex=True)
fig.subplots_adjust(hspace=0.06)
ax1, ax2, ax3 = axes

# Panel 1: Historical volatility — 3 windows
ax1.plot(df.index, df["HV_20"],   lw=1.1, color=C["vol_hist"], label="HV-20d")
ax1.plot(df.index, df["HV_60"],   lw=1.3, color=C["vol_park"], label="HV-60d", alpha=0.85)
ax1.plot(df.index, df["HV_120"],  lw=1.5, color=C["vol_gk"],   label="HV-120d", alpha=0.85)
ax1.set_ylabel("Ann. Volatility (%)", fontsize=10)
ax1.set_title("OGDC — Volatility Analysis (2020–2026)", fontsize=13, fontweight="bold", pad=10)
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# Panel 2: Three estimators on same scale
ax2.plot(df.index, df["HV_20"],           lw=1.0, color=C["vol_hist"], label="Close-Close (HV-20)")
ax2.plot(df.index, df["HV_Parkinson"],    lw=1.0, color=C["vol_park"], label="Parkinson (H-L)", alpha=0.85)
ax2.plot(df.index, df["HV_GarmanKlass"], lw=1.0, color=C["vol_gk"],  label="Garman-Klass (OHLC)", alpha=0.85)
ax2.set_ylabel("Ann. Volatility (%)", fontsize=10)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# Panel 3: Volatility regime bands
regime_colors = {"Low": C["regime_lo"], "Medium": C["regime_md"], "High": C["regime_hi"]}
for regime, color in regime_colors.items():
    mask = df["Vol_regime"] == regime
    ax3.fill_between(df.index, 0, df["HV_20"].fillna(0),
                     where=mask, color=color, alpha=0.7, label=f"{regime} Vol")
ax3.plot(df.index, df["HV_20"], lw=0.8, color="white", alpha=0.5)
ax3.axhline(q33, color=C["regime_md"], lw=1, linestyle="--", alpha=0.8)
ax3.axhline(q67, color=C["regime_hi"], lw=1, linestyle="--", alpha=0.8)
ax3.set_ylabel("HV-20 Regime", fontsize=9)
ax3.set_xlabel("Date", fontsize=9)
ax3.legend(fontsize=8, loc="upper right")
ax3.grid(True, alpha=0.3)

save_fig(fig, "04_volatility_panel.png")

# ── Chart 5: Time Series Decomposition ────────────────────────────────────────
fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)
fig.subplots_adjust(hspace=0.1)

components = [
    (price,          "Observed Price (PKR)",       C["price"]),
    (df["Trend"],    "Trend (252-day centred MA)",  C["sma200"]),
    (df["Seasonal"], "Seasonal Component",          C["ema26"]),
    (df["Residual"], "Residual",                    "#8b949e"),
]

for ax, (series, label, color) in zip(axes, components):
    if label == "Residual":
        ax.bar(series.index, series, width=1.2, color=color, alpha=0.5)
        ax.axhline(0, color="white", lw=0.7, alpha=0.5)
    else:
        ax.plot(series.index, series, lw=1.2, color=color)
        if label != "Seasonal Component":
            ax.fill_between(series.index, series, series.min(), alpha=0.1, color=color)
    ax.set_ylabel(label, fontsize=9)
    ax.grid(True, alpha=0.25)

axes[0].set_title("OGDC — Additive Time Series Decomposition", fontsize=13, fontweight="bold", pad=10)
axes[-1].set_xlabel("Date", fontsize=9)
save_fig(fig, "05_ts_decomposition.png")

# ── Chart 6: ACF / PACF ────────────────────────────────────────────────────────
lags = np.arange(0, 31)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# ACF
ax1.bar(lags, acf_vals, color=C["price"], alpha=0.8, width=0.6)
ax1.axhline( conf95, color=C["signal_sell"], lw=1.5, linestyle="--", label=f"95% CI (±{conf95:.3f})")
ax1.axhline(-conf95, color=C["signal_sell"], lw=1.5, linestyle="--")
ax1.axhline(0, color="#8b949e", lw=0.8)
ax1.set_xlabel("Lag (days)", fontsize=10)
ax1.set_ylabel("Autocorrelation", fontsize=10)
ax1.set_title("ACF — OGDC Daily Returns", fontsize=12, fontweight="bold")
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# PACF
ax2.bar(lags, pacf_vals, color=C["ema26"], alpha=0.8, width=0.6)
ax2.axhline( conf95, color=C["signal_sell"], lw=1.5, linestyle="--", label=f"95% CI (±{conf95:.3f})")
ax2.axhline(-conf95, color=C["signal_sell"], lw=1.5, linestyle="--")
ax2.axhline(0, color="#8b949e", lw=0.8)
ax2.set_xlabel("Lag (days)", fontsize=10)
ax2.set_ylabel("Partial Autocorrelation", fontsize=10)
ax2.set_title("PACF — OGDC Daily Returns", fontsize=12, fontweight="bold")
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

fig.suptitle("OGDC Return Autocorrelation Structure", fontsize=13, fontweight="bold")
plt.tight_layout()
save_fig(fig, "06_acf_pacf.png")

# ── Chart 7: Volatility vs Price — scatter by regime ─────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left: HV-20 over time coloured by regime
ax = axes[0]
for regime, color in regime_colors.items():
    mask = df["Vol_regime"] == regime
    ax.plot(df.index[mask], df["HV_20"][mask], ".", color=color, markersize=3, alpha=0.7, label=regime)
ax.set_title("Volatility Regimes Over Time", fontweight="bold")
ax.set_ylabel("HV-20 (%)")
ax.set_xlabel("Date")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Right: HV vs next-day return (absolute)
ax = axes[1]
common = df[["HV_20","Returns","Vol_regime"]].dropna()
for regime, color in regime_colors.items():
    sub_df = common[common["Vol_regime"] == regime]
    ax.scatter(sub_df["HV_20"], sub_df["Returns"].abs(),
               color=color, s=6, alpha=0.4, label=f"{regime} vol")

# Trend line
from scipy.stats import linregress
hv_v = common["HV_20"].values
rv_v = common["Returns"].abs().values
slope, intercept, r_val, *_ = linregress(hv_v, rv_v)
x_line = np.linspace(hv_v.min(), hv_v.max(), 100)
ax.plot(x_line, slope * x_line + intercept, color="white", lw=1.5, label=f"Trend (r={r_val:.2f})")
ax.set_title("HV-20 vs |Daily Return| by Regime", fontweight="bold")
ax.set_xlabel("HV-20 (%)")
ax.set_ylabel("|Return| (%)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

fig.suptitle("OGDC Volatility Analysis", fontsize=13, fontweight="bold")
plt.tight_layout()
save_fig(fig, "07_volatility_regimes.png")

# ── Chart 8: Dashboard — All Indicators Together (2024–2026 zoom) ─────────────
fig = plt.figure(figsize=(18, 14))
gs  = gridspec.GridSpec(5, 1, height_ratios=[3.5, 1, 1, 1, 1], hspace=0.05)

ax_price = fig.add_subplot(gs[0])
ax_pctb  = fig.add_subplot(gs[1], sharex=ax_price)
ax_bw    = fig.add_subplot(gs[2], sharex=ax_price)
ax_hv    = fig.add_subplot(gs[3], sharex=ax_price)
ax_ret   = fig.add_subplot(gs[4], sharex=ax_price)

zoom2    = df["2023":]
zp2      = price["2023":]

# Price + All MAs + BB
ax_price.fill_between(zoom2.index, zoom2["BB_upper"], zoom2["BB_lower"], color=C["bb_fill"], alpha=0.2)
ax_price.plot(zoom2.index, zoom2["BB_upper"], lw=0.8, color=C["bb_upper"], linestyle="--", alpha=0.7)
ax_price.plot(zoom2.index, zoom2["BB_lower"], lw=0.8, color=C["bb_lower"], linestyle="--", alpha=0.7)
ax_price.plot(zp2.index,   zp2,               lw=1.1, color=C["price"],    zorder=3, label="Price")
ax_price.plot(zoom2.index, zoom2["SMA_20"],   lw=1.0, color=C["sma20"],   label="SMA-20")
ax_price.plot(zoom2.index, zoom2["SMA_50"],   lw=1.2, color=C["sma50"],   label="SMA-50")
ax_price.plot(zoom2.index, zoom2["EMA_12"],   lw=0.8, color=C["ema12"],   linestyle=":", label="EMA-12")

# Squeeze markers
sq_z2 = zoom2["BB_squeeze"].fillna(False)
ax_price.fill_between(zoom2.index, zoom2["BB_lower"], zoom2["BB_upper"],
                      where=sq_z2, color=C["squeeze"], alpha=0.22, label="Squeeze")

ax_price.set_title("OGDC — Integrated Trend Dashboard (2023–2026)\nBollinger Bands + Moving Averages + Volatility",
                    fontsize=13, fontweight="bold", pad=8)
ax_price.set_ylabel("Price (PKR)", fontsize=9)
ax_price.legend(fontsize=7.5, ncol=5, loc="upper left")
ax_price.grid(True, alpha=0.25)

# %B
ax_pctb.plot(zoom2.index, zoom2["BB_pct_b"], lw=0.9, color="#79c0ff")
ax_pctb.axhline(0.5, color="#8b949e", lw=0.7)
ax_pctb.axhline(0,   color=C["signal_sell"], lw=0.8, linestyle="--", alpha=0.6)
ax_pctb.axhline(1,   color=C["signal_buy"],  lw=0.8, linestyle="--", alpha=0.6)
ax_pctb.fill_between(zoom2.index, zoom2["BB_pct_b"], 0.5,
                     where=zoom2["BB_pct_b"] > 0.5, color=C["signal_buy"],  alpha=0.15)
ax_pctb.fill_between(zoom2.index, zoom2["BB_pct_b"], 0.5,
                     where=zoom2["BB_pct_b"] < 0.5, color=C["signal_sell"], alpha=0.15)
ax_pctb.set_ylabel("%B", fontsize=8)
ax_pctb.set_ylim(-0.4, 1.4)
ax_pctb.grid(True, alpha=0.25)

# Bandwidth
ax_bw.plot(zoom2.index, zoom2["BB_bandwidth"] * 100, lw=0.9, color=C["squeeze"])
ax_bw.fill_between(zoom2.index, zoom2["BB_bandwidth"] * 100, 0,
                   where=sq_z2, color=C["squeeze"], alpha=0.4, label="Squeeze")
ax_bw.set_ylabel("BW%", fontsize=8)
ax_bw.grid(True, alpha=0.25)
ax_bw.legend(fontsize=7)

# HV-20
hv2 = zoom2["HV_20"]
for regime, color in regime_colors.items():
    mask = zoom2["Vol_regime"] == regime
    ax_hv.fill_between(zoom2.index, 0, hv2.fillna(0), where=mask, color=color, alpha=0.7, label=regime)
ax_hv.plot(zoom2.index, hv2, lw=0.8, color="white", alpha=0.5)
ax_hv.set_ylabel("HV-20%", fontsize=8)
ax_hv.legend(fontsize=7, ncol=3)
ax_hv.grid(True, alpha=0.25)

# Returns
ret2 = ret.reindex(zoom2.index).fillna(0)
colors_ret = [C["signal_buy"] if r >= 0 else C["signal_sell"] for r in ret2]
ax_ret.bar(zoom2.index, ret2, width=1.2, color=colors_ret, alpha=0.8)
ax_ret.axhline(0, color="#8b949e", lw=0.7)
ax_ret.set_ylabel("Return %", fontsize=8)
ax_ret.set_xlabel("Date", fontsize=9)
ax_ret.grid(True, alpha=0.25)

plt.setp(ax_price.get_xticklabels(), visible=False)
plt.setp(ax_pctb.get_xticklabels(),  visible=False)
plt.setp(ax_bw.get_xticklabels(),    visible=False)
plt.setp(ax_hv.get_xticklabels(),    visible=False)
save_fig(fig, "08_integrated_dashboard.png")

# ══════════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════════
banner("Summary")

print(f"""
  ┌─────────────────────────────────────────────────────────────────┐
  │  OGDC Financial Market Trend Analysis — Summary                 │
  ├───────────────────────────┬─────────────────────────────────────┤
  │ MOVING AVERAGES           │                                     │
  │  SMA 20/50/200            │ Computed over full 2020–2026 period │
  │  EMA 12/26                │ Exponentially weighted              │
  │  WMA 20                   │ Linearly weighted                   │
  │  Golden Crosses           │ {len(golden_crosses):>4} detected   │
  │  Death Crosses            │ {len(death_crosses):>4} detected    │
  ├───────────────────────────┼─────────────────────────────────────┤
  │ BOLLINGER BANDS           │                                     │
  │  Window / σ               │ 20-day / 2σ                         │
  │  Squeeze days             │ {squeeze_days:>4}                   │
  │  Breakout-long signals    │ {bo_long_days:>4}                   │
  │  Breakout-short signals   │ {bo_shrt_days:>4}                   │
  │  MR-Buy signals           │ {mr_buy_days:>4}                    │
  │  MR-Sell signals          │ {mr_sell_days:>4}                   │
  ├───────────────────────────┼─────────────────────────────────────┤
  │ VOLATILITY                │                                     │
  │  HV-20 current            │ {df["HV_20"].iloc[-1]:.1f}%         │
  │  HV-20 average            │ {df["HV_20"].mean():.1f}%           │
  │  Regime thresholds        │ Low<{q33:.1f}%  High>{q67:.1f}%     │
  │  ARCH effects             │ {'Present (clustering expected)' if 
                                 p_arch < 0.05 else 'Not detected'} │
  ├───────────────────────────┼─────────────────────────────────────┤
  │ TIME SERIES               │                                     │
  │  Price stationarity       │ Non-stationary (unit root)          │
  │  Returns stationarity     │ Stationary (yes)                    │
  │  Sig. ACF lags            │ {sig_acf.tolist()}                  
  │  Sig. PACF lags           │ {sig_pacf.tolist()}                 │
  ├───────────────────────────┼─────────────────────────────────────┤
  │ OUTPUTS                   │                                     │
  │  ogdc_trend_features.csv  │ {len(feature_df.columns)} features 
                              × {len(feature_df)} rows              │
  │  trend_01 to trend_08.png │ 8 charts                            │
  └───────────────────────────┴─────────────────────────────────────┘
""")
