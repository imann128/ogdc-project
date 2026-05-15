"""
ogdc_garch.py
=============
GARCH Volatility Modelling for OGDC — PSX

Models implemented using scipy.optimize:
  1. ARCH(1)     — baseline, volatility depends only on last shock
  2. GARCH(1,1)  — industry standard, adds persistence term
  3. EGARCH(1,1) — asymmetric: negative shocks inflate vol more than positive
  4. GJR-GARCH   — Glosten-Jagannathan-Runkle: threshold asymmetry

For each model:
  • MLE parameter estimation
  • AIC / BIC model selection
  • Conditional volatility series
  • 30-day out-of-sample forecast
  • Volatility regime classification on forecast path

Outputs: 5 PNG charts + ogdc_garch_results.csv
"""
import sys, os, shutil, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from paths import processed, img, frontend, PATHS, ensure_dirs
ensure_dirs()

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE      = os.path.dirname(os.path.abspath(__file__)) # Scripts
# Data Directory
DATA_DIR = os.path.join(BASE, "..", "data", "processed")  
os.makedirs(DATA_DIR, exist_ok=True)
DATA_PATH = os.path.join(DATA_DIR, "ogdc_trend_features.csv")  
# Output Directory
OUTPUT_DIR = os.path.join(BASE, "..", "data", "processed")
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUT_CSV = os.path.join(OUTPUT_DIR, "ogdc_garch_results.csv") 
# Image Directory
IMG_DIR = os.path.join(BASE, "..", "outputs", "images")  
os.makedirs(IMG_DIR, exist_ok=True)
IMG_PRE = os.path.join(IMG_DIR, "garch_")  

import warnings, os
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats, optimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

# Themes
plt.rcParams.update({
    "figure.facecolor":"#0d1117","axes.facecolor":"#161b22",
    "axes.edgecolor":"#30363d","axes.labelcolor":"#e6edf3",
    "xtick.color":"#8b949e","ytick.color":"#8b949e",
    "text.color":"#e6edf3","grid.color":"#21262d","grid.linewidth":0.6,
    "legend.facecolor":"#161b22","legend.edgecolor":"#30363d",
})
C = {"ret":"#58a6ff","vol":"#f0883e","egarch":"#3fb950","gjr":"#d2a8ff",
     "forecast":"#ffa657","ci":"#1f4080","hi":"#f85149","lo":"#2ea043","md":"#d29922"}

TRADING_DAYS = 252

# helper function to save figures to image directory
def save_fig(fig, name):
    path = IMG_PRE + name
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  → saved: garch_{name}")

# helper functions for formatted console output
def banner(t): print(f"\n{'═'*70}\n  {t}\n{'═'*70}")
def sub(t):    print(f"\n  {'─'*58}\n  {t}\n  {'─'*58}")

# ══════════════════════════════════════════════════════════════════════════════
# Load Data
# ══════════════════════════════════════════════════════════════════════════════
banner("Loading Data")
df   = pd.read_csv(DATA_PATH, index_col="Date", parse_dates=True)
ret  = df["Returns"].dropna()                         # daily % returns
logr = np.log(df["Price"] / df["Price"].shift(1)).dropna() * 100   # log returns %

# Demean returns for GARCH (model the variance of zero-mean series)
r = logr.values
mu_r = r.mean()
e = r - mu_r          # demeaned innovations

n = len(e)
print(f"  Observations : {n}")
print(f"  Mean return  : {mu_r:.4f}%")
print(f"  Std return   : {e.std():.4f}%")
print(f"  Skewness     : {stats.skew(e):.4f}")
print(f"  Kurtosis     : {stats.kurtosis(e):.4f}  (excess)")

# ══════════════════════════════════════════════════════════════════════════════
# Model Definitons
# ══════════════════════════════════════════════════════════════════════════════

# ── Negative log-likelihood functions ─────────────────────────────────────────
def negll_arch1(params, e):
    omega, alpha = params
    if omega <= 0 or alpha < 0 or alpha >= 1:
        return 1e9
    n = len(e)
    h = np.full(n, omega / (1 - alpha + 1e-8))
    for t in range(1, n):
        h[t] = omega + alpha * e[t-1]**2
        h[t] = max(h[t], 1e-8)
    ll = -0.5 * np.sum(np.log(2*np.pi) + np.log(h) + e**2 / h)
    return -ll

# negative log-likelihood for garch(1,1) model
def negll_garch11(params, e):
    omega, alpha, beta = params
    if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
        return 1e9
    n = len(e)
    h = np.full(n, omega / (1 - alpha - beta + 1e-8))
    for t in range(1, n):
        h[t] = omega + alpha * e[t-1]**2 + beta * h[t-1]
        h[t] = max(h[t], 1e-8)
    ll = -0.5 * np.sum(np.log(2*np.pi) + np.log(h) + e**2 / h)
    return -ll

#  negative log-likelihood for egarch(1,1) with leverage term
def negll_egarch(params, e):
    omega, alpha, gamma, beta = params
    if abs(beta) >= 1:
        return 1e9
    n = len(e)
    log_h = np.full(n, omega / (1 - beta + 1e-8))
    for t in range(1, n):
        z = e[t-1] / np.exp(0.5 * log_h[t-1])
        log_h[t] = omega + beta * log_h[t-1] + alpha * z + gamma * (abs(z) - np.sqrt(2/np.pi))
    h = np.exp(log_h)
    h = np.clip(h, 1e-8, 1e8)
    ll = -0.5 * np.sum(np.log(2*np.pi) + log_h + e**2 / h)
    return -ll

#  negative log-likelihood for gjr-garch with threshold asymmetry
def negll_gjr(params, e):
    omega, alpha, gamma, beta = params
    if omega <= 0 or alpha < 0 or beta < 0 or alpha + 0.5*gamma + beta >= 1:
        return 1e9
    n = len(e)
    h = np.full(n, omega / (1 - alpha - 0.5*gamma - beta + 1e-8))
    for t in range(1, n):
        ind = 1.0 if e[t-1] < 0 else 0.0   # leverage indicator
        h[t] = omega + alpha * e[t-1]**2 + gamma * ind * e[t-1]**2 + beta * h[t-1]
        h[t] = max(h[t], 1e-8)
    ll = -0.5 * np.sum(np.log(2*np.pi) + np.log(h) + e**2 / h)
    return -ll

# ── Conditional variance extractors ───────────────────────────────────────────
# extract conditional variance from fitted garch(1,1)
def extract_garch11(params, e):
    omega, alpha, beta = params
    n = len(e)
    h = np.full(n, omega / max(1 - alpha - beta, 1e-8))
    for t in range(1, n):
        h[t] = max(omega + alpha * e[t-1]**2 + beta * h[t-1], 1e-8)
    return h

#  extract conditional variance from fitted egarch
def extract_egarch(params, e):
    omega, alpha, gamma, beta = params
    n = len(e)
    log_h = np.full(n, omega / max(1 - beta, 1e-8))
    for t in range(1, n):
        z = e[t-1] / np.exp(0.5 * log_h[t-1])
        log_h[t] = omega + beta * log_h[t-1] + alpha * z + gamma * (abs(z) - np.sqrt(2/np.pi))
    return np.clip(np.exp(log_h), 1e-8, 1e8)

# extract conditional variance from fitted gjr-garch 
def extract_gjr(params, e):
    omega, alpha, gamma, beta = params
    n = len(e)
    h = np.full(n, omega / max(1 - alpha - 0.5*gamma - beta, 1e-8))
    for t in range(1, n):
        ind = 1.0 if e[t-1] < 0 else 0.0
        h[t] = max(omega + alpha * e[t-1]**2 + gamma * ind * e[t-1]**2 + beta * h[t-1], 1e-8)
    return h

# calculate akaike and bayesian information criteria
def aic_bic(negll_val, k, n):
    ll = -negll_val
    return 2*k - 2*ll, np.log(n)*k - 2*ll

# ══════════════════════════════════════════════════════════════════════════════
# Fit all models
# ══════════════════════════════════════════════════════════════════════════════
banner("Fitting Garch Models")

# Variance of residuals as starting point
var0 = np.var(e)

models_fit = {}

# ── ARCH(1) ───────────────────────────────────────────────────────────────────
#  fit arch(1) model using lbfgs-b optimizer
sub("ARCH(1)")
res_arch = optimize.minimize(
    negll_arch1, x0=[var0 * 0.1, 0.2], args=(e,),
    method="L-BFGS-B",
    bounds=[(1e-6, None), (1e-6, 0.999)],
    options={"maxiter": 2000}
)
p_arch = res_arch.x
aic_a, bic_a = aic_bic(res_arch.fun, 2, n)
h_arch = np.array([max(p_arch[0] + p_arch[1]*e[max(0,t-1)]**2, 1e-8) for t in range(n)])
vol_arch = np.sqrt(h_arch * TRADING_DAYS)   # annualised %

print(f"  ω={p_arch[0]:.6f}  α={p_arch[1]:.4f}")
print(f"  AIC={aic_a:.2f}  BIC={bic_a:.2f}")
models_fit["ARCH(1)"] = {"params": p_arch, "aic": aic_a, "bic": bic_a, "vol": vol_arch, "k":2}

# ── GARCH(1,1) ────────────────────────────────────────────────────────────────
# fit garch(1,1) model 
sub("GARCH(1,1)")
res_g11 = optimize.minimize(
    negll_garch11, x0=[var0*0.05, 0.1, 0.85], args=(e,),
    method="L-BFGS-B",
    bounds=[(1e-6, None), (1e-6, 0.999), (1e-6, 0.999)],
    options={"maxiter": 3000}
)
p_g11 = res_g11.x
aic_g, bic_g = aic_bic(res_g11.fun, 3, n)
h_g11  = extract_garch11(p_g11, e)
vol_g11 = np.sqrt(h_g11 * TRADING_DAYS)

omega_g, alpha_g, beta_g = p_g11
persistence = alpha_g + beta_g
half_life   = np.log(0.5) / np.log(persistence) if 0 < persistence < 1 else np.inf
unconditional_vol = np.sqrt(omega_g / max(1 - persistence, 1e-8) * TRADING_DAYS)

print(f"  ω={omega_g:.6f}  α={alpha_g:.4f}  β={beta_g:.4f}")
print(f"  Persistence (α+β): {persistence:.4f}")
print(f"  Half-life of shock: {half_life:.1f} days")
print(f"  Long-run (unconditional) vol: {unconditional_vol:.2f}%")
print(f"  AIC={aic_g:.2f}  BIC={bic_g:.2f}")
models_fit["GARCH(1,1)"] = {"params": p_g11, "aic": aic_g, "bic": bic_g,
                              "vol": vol_g11, "k":3,
                              "persistence": persistence, "half_life": half_life,
                              "unconditional_vol": unconditional_vol}

# ── EGARCH(1,1) ───────────────────────────────────────────────────────────────
# fit egarch(1,1) with nelder-mead
sub("EGARCH(1,1) — Asymmetric")
res_eg = optimize.minimize(
    negll_egarch, x0=[np.log(var0)*0.05, 0.1, -0.1, 0.85], args=(e,),
    method="Nelder-Mead",
    options={"maxiter": 10000, "xatol":1e-6, "fatol":1e-6}
)
p_eg = res_eg.x
aic_e, bic_e = aic_bic(res_eg.fun, 4, n)
h_eg   = extract_egarch(p_eg, e)
vol_eg = np.sqrt(h_eg * TRADING_DAYS)

print(f"  ω={p_eg[0]:.4f}  α={p_eg[1]:.4f}  γ={p_eg[2]:.4f}  β={p_eg[3]:.4f}")
print(f"  Leverage (γ): {p_eg[2]:.4f}  → {'Negative shock ↑ vol more' if p_eg[2] < 0 else 'Symmetric or inverse'}")
print(f"  AIC={aic_e:.2f}  BIC={bic_e:.2f}")
models_fit["EGARCH(1,1)"] = {"params": p_eg, "aic": aic_e, "bic": bic_e, "vol": vol_eg, "k":4}

# ── GJR-GARCH(1,1) ────────────────────────────────────────────────────────────
#  fit gjr-garch with threshold leverage term
sub("GJR-GARCH(1,1) — Threshold Asymmetry")
res_gjr = optimize.minimize(
    negll_gjr, x0=[var0*0.05, 0.05, 0.1, 0.85], args=(e,),
    method="L-BFGS-B",
    bounds=[(1e-6,None),(1e-6,0.999),(0,0.999),(1e-6,0.999)],
    options={"maxiter": 3000}
)
p_gjr = res_gjr.x
aic_gjr, bic_gjr = aic_bic(res_gjr.fun, 4, n)
h_gjr   = extract_gjr(p_gjr, e)
vol_gjr = np.sqrt(h_gjr * TRADING_DAYS)

print(f"  ω={p_gjr[0]:.6f}  α={p_gjr[1]:.4f}  γ={p_gjr[2]:.4f}  β={p_gjr[3]:.4f}")
print(f"  Leverage (γ): {p_gjr[2]:.4f}  → {'Asymmetry confirmed' if p_gjr[2] > 0 else 'No threshold effect'}")
print(f"  AIC={aic_gjr:.2f}  BIC={bic_gjr:.2f}")
models_fit["GJR-GARCH(1,1)"] = {"params": p_gjr, "aic": aic_gjr, "bic": bic_gjr,
                                   "vol": vol_gjr, "k":4}

# ── Model selection ────────────────────────────────────────────────────────────
# compare all models using aic and bic, select best 
sub("Model Selection (AIC / BIC)")
sel = pd.DataFrame({m: {"AIC": v["aic"], "BIC": v["bic"]} for m,v in models_fit.items()}).T
sel["Rank_AIC"] = sel["AIC"].rank().astype(int)
sel["Rank_BIC"] = sel["BIC"].rank().astype(int)
print(f"\n{sel.round(2).to_string()}")
best_model = sel["AIC"].idxmin()
print(f"\n  Best model by AIC: {best_model}")

# ══════════════════════════════════════════════════════════════════════════════
# 30-Day Forecast — GARCH(1,1) (most interpretable; best for forecasting)
# ══════════════════════════════════════════════════════════════════════════════
banner("30-Day Volatility Forecast — GARCH(1,1)")

HORIZON = 30
omega_f, alpha_f, beta_f = p_g11
h_last   = h_g11[-1]
e_last   = e[-1]

# Multi-step GARCH forecast: E[h_{t+k}] = ω·Σ(α+β)^j + (α+β)^k · h_t
h_fcast  = np.zeros(HORIZON)
h_fcast[0] = omega_f + alpha_f * e_last**2 + beta_f * h_last
for k in range(1, HORIZON):
    h_fcast[k] = omega_f + (alpha_f + beta_f) * h_fcast[k-1]

vol_fcast    = np.sqrt(h_fcast * TRADING_DAYS)   # annualised %
vol_fcast_d  = np.sqrt(h_fcast)                  # daily %

# 95% forecast interval (approximate, normal)
ci_upper = vol_fcast * 1.645
ci_lower = np.maximum(vol_fcast * 0.355, 0)

last_date  = logr.index[-1]
fcast_idx  = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=HORIZON)

# Volatility regime on forecast path
# classify forecast into volatility regimes (low/medium/high)
q33_f, q67_f = np.percentile(vol_g11, [33, 67])
fcast_regimes = ["High" if v > q67_f else ("Low" if v < q33_f else "Medium")
                 for v in vol_fcast]

print(f"\n  GARCH(1,1) 30-day forecast:")
print(f"  {'Day':>4}  {'Date':12}  {'Ann.Vol%':>9}  {'Daily Vol%':>10}  {'Regime':8}")
print(f"  {'─'*52}")
for i in range(HORIZON):
    print(f"  {i+1:>4}  {str(fcast_idx[i].date()):12}  "
          f"{vol_fcast[i]:>9.2f}  {vol_fcast_d[i]:>10.4f}  {fcast_regimes[i]:8}")

print(f"\n  Current GARCH(1,1) vol : {vol_g11[-1]:.2f}% (annualised)")
print(f"  30-day terminal vol    : {vol_fcast[-1]:.2f}%")
print(f"  Long-run mean-reversion: {unconditional_vol:.2f}%")
direction = "⬇ reverting down" if vol_g11[-1] > unconditional_vol else "⬆ reverting up"
print(f"  Direction              : {direction}")

# ══════════════════════════════════════════════════════════════════════════════
# Save Results
# ══════════════════════════════════════════════════════════════════════════════
banner("Saving Results")

# In-sample conditional volatilities
# create dataframe with in-sample conditional volatilities
result_df = pd.DataFrame({
    "Returns":             logr.values,
    "Vol_ARCH1":           vol_arch,
    "Vol_GARCH11":         vol_g11,
    "Vol_EGARCH":          vol_eg,
    "Vol_GJRGARCH":        vol_gjr,
}, index=logr.index)

# Forecast rows
#  create forecast dataframe with confidence intervals and regimes
fcast_df = pd.DataFrame({
    "Returns":      np.nan,
    "Vol_GARCH11":  vol_fcast,
    "Vol_CI_Upper": ci_upper,
    "Vol_CI_Lower": ci_lower,
    "Forecast_Day": range(1, HORIZON+1),
    "Regime":       fcast_regimes,
}, index=fcast_idx)

result_df.to_csv(OUT_CSV)
print(f"  Saved {len(result_df)} in-sample rows → ogdc_garch_results.csv")

# ══════════════════════════════════════════════════════════════════════════════
# Visualizations
# ══════════════════════════════════════════════════════════════════════════════
banner("Generating Charts")

# ── Chart 1: All 4 conditional volatilities ────────────────────────────────────
fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)
fig.subplots_adjust(hspace=0.06)
pairs = [
    ("Vol_ARCH1",    "ARCH(1)",       C["ret"]),
    ("Vol_GARCH11",  "GARCH(1,1)",    C["vol"]),
    ("Vol_EGARCH",   "EGARCH(1,1)",   C["egarch"]),
    ("Vol_GJRGARCH", "GJR-GARCH(1,1)",C["gjr"]),
]
for ax, (col, label, color) in zip(axes, pairs):
    ax.plot(result_df.index, result_df[col], lw=0.9, color=color, label=label)
    ax.fill_between(result_df.index, result_df[col], alpha=0.15, color=color)
    ax.axhline(result_df[col].mean(), color="white", lw=0.8, linestyle="--", alpha=0.5,
               label=f"Mean={result_df[col].mean():.1f}%")
    ax.set_ylabel("Ann. Vol %", fontsize=9)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.25)

axes[0].set_title("OGDC — Conditional Volatility: All GARCH Models (2020–2026)",
                   fontsize=13, fontweight="bold", pad=10)
axes[-1].set_xlabel("Date", fontsize=9)
save_fig(fig, "01_all_models.png")

# ── Chart 2: GARCH(1,1) vs EGARCH vs GJR — leverage comparison ────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 9), sharex=True)
fig.subplots_adjust(hspace=0.08)

ax1.plot(result_df.index, result_df["Vol_GARCH11"],  lw=1.0, color=C["vol"],    label="GARCH(1,1) — Symmetric")
ax1.plot(result_df.index, result_df["Vol_EGARCH"],   lw=0.9, color=C["egarch"], label="EGARCH(1,1) — Asymmetric", alpha=0.85)
ax1.plot(result_df.index, result_df["Vol_GJRGARCH"], lw=0.9, color=C["gjr"],    label="GJR-GARCH — Threshold", alpha=0.85)
ax1.axhline(unconditional_vol, color="white", lw=1, linestyle=":", alpha=0.6,
            label=f"Long-run vol ({unconditional_vol:.1f}%)")
ax1.set_ylabel("Ann. Volatility %", fontsize=10)
ax1.set_title("GARCH Model Comparison — Symmetric vs Asymmetric Specifications",
               fontsize=13, fontweight="bold", pad=10)
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.25)

# Difference: EGARCH - GARCH (leverage effect)
diff = result_df["Vol_EGARCH"] - result_df["Vol_GARCH11"]
ax2.plot(diff.index, diff, lw=0.9, color=C["gjr"], alpha=0.8)
ax2.fill_between(diff.index, diff, 0,
                 where=diff > 0, color=C["hi"], alpha=0.3, label="EGARCH > GARCH (neg shock amplified)")
ax2.fill_between(diff.index, diff, 0,
                 where=diff < 0, color=C["lo"], alpha=0.3, label="EGARCH < GARCH")
ax2.axhline(0, color="#8b949e", lw=0.8)
ax2.set_ylabel("EGARCH − GARCH (pp)", fontsize=9)
ax2.set_xlabel("Date", fontsize=9)
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.25)
save_fig(fig, "02_leverage_comparison.png")

# ── Chart 3: 30-day forecast ──────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 9))
fig.subplots_adjust(hspace=0.12)

# Historical (last 120 days) + forecast
hist_window = result_df["Vol_GARCH11"].iloc[-120:]
ax1.plot(hist_window.index, hist_window, lw=1.2, color=C["vol"], label="Historical GARCH(1,1) Vol")
ax1.plot(fcast_idx, vol_fcast, lw=1.5, color=C["forecast"], label="30-day Forecast", linestyle="--")
ax1.fill_between(fcast_idx, ci_lower, ci_upper, color=C["ci"], alpha=0.4, label="95% Confidence Band")
ax1.axvline(last_date, color="white", lw=1, linestyle=":", alpha=0.7)
ax1.axhline(unconditional_vol, color="#8b949e", lw=1, linestyle=":", label=f"Long-run Vol ({unconditional_vol:.1f}%)")

# Colour forecast by regime
for i, (date, regime) in enumerate(zip(fcast_idx, fcast_regimes)):
    col = C["hi"] if regime=="High" else (C["lo"] if regime=="Low" else C["md"])
    ax1.axvspan(date - pd.Timedelta(days=0.5), date + pd.Timedelta(days=0.5),
                color=col, alpha=0.12)

ax1.set_title("OGDC — GARCH(1,1) 30-Day Volatility Forecast", fontsize=13, fontweight="bold", pad=10)
ax1.set_ylabel("Ann. Volatility %", fontsize=10)
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.25)

# Daily forecast bar
bar_colors = [C["hi"] if r=="High" else (C["lo"] if r=="Low" else C["md"])
              for r in fcast_regimes]
bars = ax2.bar(range(1, HORIZON+1), vol_fcast, color=bar_colors, alpha=0.85, edgecolor="#0d1117", width=0.7)
ax2.plot(range(1, HORIZON+1), ci_upper, lw=1, color="white", linestyle="--", alpha=0.5, label="95% CI Upper")
ax2.plot(range(1, HORIZON+1), ci_lower, lw=1, color="white", linestyle="--", alpha=0.5, label="95% CI Lower")
ax2.axhline(unconditional_vol, color="#8b949e", lw=1, linestyle=":", label=f"LR vol={unconditional_vol:.1f}%")
hi_p = mpatches.Patch(color=C["hi"], label="High Vol Regime")
md_p = mpatches.Patch(color=C["md"], label="Medium Vol Regime")
lo_p = mpatches.Patch(color=C["lo"], label="Low Vol Regime")
ax2.legend(handles=[hi_p, md_p, lo_p], fontsize=8, loc="upper right")
ax2.set_xlabel("Forecast Horizon (Trading Days)", fontsize=10)
ax2.set_ylabel("Forecasted Ann. Vol %", fontsize=10)
ax2.set_title("30-Day Forecast Profile — by Volatility Regime", fontsize=11, fontweight="bold")
ax2.grid(True, alpha=0.25, axis="y")

save_fig(fig, "03_forecast.png")

# ── Chart 4: News impact curve (GARCH vs EGARCH) ─────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
z_range = np.linspace(-4, 4, 300)
h_base  = np.var(e)

# GARCH news impact: h_{t+1} = ω + α·z²·h_base + β·h_base
nic_garch = omega_g + alpha_g * z_range**2 * h_base + beta_g * h_base
ax1.plot(z_range, np.sqrt(nic_garch * TRADING_DAYS),
         lw=2, color=C["vol"], label="GARCH(1,1) — Symmetric")
# EGARCH news impact
nic_egarch = []
for z in z_range:
    log_h_next = p_eg[0] + p_eg[3] * np.log(h_base) + p_eg[1]*z + p_eg[2]*(abs(z) - np.sqrt(2/np.pi))
    nic_egarch.append(np.sqrt(np.exp(log_h_next) * TRADING_DAYS))
ax1.plot(z_range, nic_egarch, lw=2, color=C["egarch"], linestyle="--", label="EGARCH — Asymmetric")
ax1.axvline(0, color="#8b949e", lw=0.8)
ax1.set_xlabel("Standardised Shock (z)", fontsize=10)
ax1.set_ylabel("Next-Period Ann. Vol %", fontsize=10)
ax1.set_title("News Impact Curve\n(How shocks affect next-period volatility)", fontweight="bold")
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# GJR news impact
nic_gjr_pos = p_gjr[0] + p_gjr[1] * z_range**2 * h_base + p_gjr[3] * h_base
nic_gjr_neg = p_gjr[0] + (p_gjr[1]+p_gjr[2]) * z_range**2 * h_base + p_gjr[3] * h_base
ax2.plot(z_range[z_range >= 0], np.sqrt(nic_gjr_pos[z_range >= 0] * TRADING_DAYS),
         lw=2, color=C["lo"], label="GJR-GARCH: Positive shock")
ax2.plot(z_range[z_range <  0], np.sqrt(nic_gjr_neg[z_range <  0] * TRADING_DAYS),
         lw=2, color=C["hi"], label="GJR-GARCH: Negative shock (leverage)")
ax2.plot(z_range, np.sqrt(nic_garch * TRADING_DAYS),
         lw=1.5, color=C["vol"], linestyle=":", alpha=0.7, label="GARCH reference")
ax2.axvline(0, color="#8b949e", lw=0.8)
ax2.set_xlabel("Standardised Shock (z)", fontsize=10)
ax2.set_ylabel("Next-Period Ann. Vol %", fontsize=10)
ax2.set_title("GJR-GARCH News Impact Curve\n(Threshold leverage effect)", fontweight="bold")
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

fig.suptitle("OGDC — News Impact Analysis", fontsize=13, fontweight="bold")
plt.tight_layout()
save_fig(fig, "04_news_impact.png")

# ── Chart 5: Standardised residuals diagnostics ────────────────────────────────
std_resid = e / np.sqrt(h_g11)

fig = plt.figure(figsize=(16, 10))
gs  = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)
ax1 = fig.add_subplot(gs[0, :])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[1, 1])

ax1.plot(logr.index, std_resid, lw=0.7, color=C["ret"], alpha=0.8)
ax1.axhline(0, color="#8b949e", lw=0.8)
ax1.axhline( 2, color=C["hi"], lw=1, linestyle="--", alpha=0.7)
ax1.axhline(-2, color=C["hi"], lw=1, linestyle="--", alpha=0.7)
ax1.fill_between(logr.index, std_resid, 0,
                 where=std_resid > 2,  color=C["hi"], alpha=0.3)
ax1.fill_between(logr.index, std_resid, 0,
                 where=std_resid < -2, color=C["hi"], alpha=0.3)
n_outliers = ((std_resid > 2) | (std_resid < -2)).sum()
ax1.set_title(f"GARCH(1,1) Standardised Residuals  |  Outliers (|z|>2): {n_outliers}",
               fontsize=12, fontweight="bold")
ax1.set_ylabel("z-score", fontsize=9)
ax1.grid(True, alpha=0.25)

ax2.hist(std_resid, bins=50, color=C["ret"], edgecolor="#0d1117", alpha=0.85, density=True)
x_n = np.linspace(-5, 5, 200)
ax2.plot(x_n, stats.norm.pdf(x_n), color="white", lw=1.5, label="N(0,1)")
ax2.plot(x_n, stats.t.pdf(x_n, df=5), color=C["forecast"], lw=1.5, linestyle="--", label="t(5)")
ax2.set_title("Residuals Distribution", fontweight="bold")
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.25)

(osm, osr), (slope, intercept, _) = stats.probplot(std_resid, dist="norm")
ax3.scatter(osm, osr, s=5, alpha=0.5, color=C["ret"])
ax3.plot(osm, slope*np.array(osm)+intercept, color="white", lw=1.5)
ax3.set_title("Q-Q Plot vs Normal", fontweight="bold")
ax3.set_xlabel("Theoretical Quantiles")
ax3.set_ylabel("Sample Quantiles")
ax3.grid(True, alpha=0.25)

fig.suptitle("OGDC GARCH(1,1) — Residual Diagnostics", fontsize=13, fontweight="bold")
save_fig(fig, "05_diagnostics.png")

banner("Summary")
print(f"""
  ┌────────────────────────────────────────────────────────────────┐
  │  GARCH Modelling Summary                                       │
  ├──────────────────────┬─────────────────────────────────────────┤
  │ Best model (AIC)     │ {best_model:<40}│
  │ GARCH(1,1) α+β       │ {persistence:.4f}  (near 1 = strong persistence)  │
  │ Shock half-life      │ {half_life:.1f} trading days                     │
  │ Long-run vol         │ {unconditional_vol:.2f}% annualised                     │
  │ Current vol (GARCH)  │ {vol_g11[-1]:.2f}% annualised                     │
  │ 30-day fcast vol     │ {vol_fcast[-1]:.2f}% (day 30)                     │
  │ Leverage effect (γ)  │ GJR γ={p_gjr[2]:.4f}  EGARCH γ={p_eg[2]:.4f}       │
  │ Outlier days (|z|>2) │ {n_outliers:<40}│
  └──────────────────────┴─────────────────────────────────────────┘
""")
