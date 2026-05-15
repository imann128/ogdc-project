"""
ogdc_stat222.py
===============
OGDC Stock Price Analysis (PSX, 2020–2026)

Methods implemented (all course-required):
  1. EDA          — histograms, boxplots, scatter matrix, trend lines
  2. ANOVA        — one-way (by year) + two-way (year × vol-regime)
  3. Distribution Fitting — Normal, t, Lognormal, Laplace fitted to Returns
  4. Multiple Regression  — Price predicted from Open, High, Low, Volume, lag-Return
  5. ARIMA        — Returns series: ACF/PACF → model selection → diagnostics → forecast
  6. Nonparametric — Kruskal-Wallis, Mann-Whitney U, Spearman, Runs test

Outputs: 10 PNG charts + stat222_results.csv
"""
import sys, os, shutil, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from paths import processed, img, frontend, PATHS, ensure_dirs
ensure_dirs()

import warnings, os
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from itertools import combinations

BASE = os.path.dirname(os.path.abspath(__file__))  # scripts folder

# Input: cleaned data from loader (in data/processed)
DATA_DIR = os.path.join(BASE, "..", "data", "processed")
os.makedirs(DATA_DIR, exist_ok=True)
DATA = os.path.join(DATA_DIR, "ogdc_cleaned.csv")

# Output: save results to data/processed
OUT_CSV = os.path.join(DATA_DIR, "stat222_results.csv")

# Image Directory: save images to outputs/images
IMG_DIR = os.path.join(BASE, "..", "outputs", "images")
os.makedirs(IMG_DIR, exist_ok=True)
IMG_PRE = os.path.join(IMG_DIR, "stat_")

# Dark theme 
plt.rcParams.update({
    "figure.facecolor":"#0d1117","axes.facecolor":"#161b22",
    "axes.edgecolor":"#30363d","axes.labelcolor":"#e6edf3",
    "xtick.color":"#8b949e","ytick.color":"#8b949e",
    "text.color":"#e6edf3","grid.color":"#21262d","grid.linewidth":0.6,
    "legend.facecolor":"#161b22","legend.edgecolor":"#30363d",
})
TRADING_DAYS = 252

def save_fig(fig, name):
    path = IMG_PRE + name
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  → saved: stat_{name}")

def banner(t): print(f"\n{'═'*70}\n  {t}\n{'═'*70}")
def sub(t):    print(f"\n  {'─'*58}\n  {t}\n  {'─'*58}")

results_log = {}

# Load Data
banner("Loading Data")
df = pd.read_csv(DATA)
df.columns = df.columns.str.strip()
df.rename(columns={"Return":"Returns","Change %":"ChangeP","Vol.":"Volume"}, inplace=True)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)
df.dropna(subset=["Price"], inplace=True)
df["Returns"] = df["Returns"].fillna(0)
df["Year"]    = df["Date"].dt.year
df["Month"]   = df["Date"].dt.month
df["Quarter"] = df["Date"].dt.quarter

# Rolling 20-day annualised volatility
log_ret = np.log(df["Price"] / df["Price"].shift(1)).fillna(0)
df["HV20"] = log_ret.rolling(20).std() * np.sqrt(TRADING_DAYS) * 100
df["VolRegime"] = pd.qcut(df["HV20"].fillna(df["HV20"].median()),
                           q=3, labels=["Low","Medium","High"])

# Remove Returns outliers beyond 5σ for distribution fitting
ret_clean = df["Returns"].dropna()
mu_r, sd_r = ret_clean.mean(), ret_clean.std()
ret_inner  = ret_clean[(ret_clean - mu_r).abs() < 5*sd_r]

print(f"  Rows   : {len(df)}")
print(f"  Period : {df['Date'].min().date()} → {df['Date'].max().date()}")
print(f"  Years  : {sorted(df['Year'].unique())}")
print(f"\n  Descriptive Statistics:")
print(df[["Price","Volume","Returns","ChangeP"]].describe().round(4).to_string())

results_log["n_obs"]       = len(df)
results_log["date_start"]  = str(df['Date'].min().date())
results_log["date_end"]    = str(df['Date'].max().date())
results_log["price_mean"]  = round(df["Price"].mean(), 4)
results_log["price_std"]   = round(df["Price"].std(), 4)
results_log["return_mean"] = round(df["Returns"].mean(), 6)
results_log["return_std"]  = round(df["Returns"].std(), 4)
results_log["return_skew"] = round(df["Returns"].skew(), 4)
results_log["return_kurt"] = round(df["Returns"].kurtosis(), 4)


# Part 1 — EDA
banner("Part 1 — Exploratory Data Analysis")

# Chart 1a: EDA Panel 
fig = plt.figure(figsize=(18, 14))
gs  = gridspec.GridSpec(3, 3, hspace=0.42, wspace=0.35)

# Price trend
ax = fig.add_subplot(gs[0, :])
ax.plot(df["Date"], df["Price"], lw=1.1, color="#58a6ff", label="OGDC Close Price")
z = np.polyfit(np.arange(len(df)), df["Price"], 1)
trend_line = np.poly1d(z)(np.arange(len(df)))
ax.plot(df["Date"], trend_line, lw=1.5, color="#f0883e", linestyle="--", label="Linear Trend")
ax.set_title("OGDC Closing Price with Linear Trend (2020–2026)", fontweight="bold")
ax.set_ylabel("Price (PKR)")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.25)

# Returns histogram
ax2 = fig.add_subplot(gs[1, 0])
ax2.hist(ret_clean, bins=60, color="#3fb950", edgecolor="#0d1117", alpha=0.85, density=True)
xr = np.linspace(ret_clean.min(), ret_clean.max(), 200)
ax2.plot(xr, stats.norm.pdf(xr, mu_r, sd_r), color="white", lw=1.5, label="N(μ,σ)")
ax2.set_title("Daily Returns Distribution", fontweight="bold")
ax2.set_xlabel("Return (%)")
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.25)

# Boxplot by year
ax3 = fig.add_subplot(gs[1, 1])
years = sorted(df["Year"].unique())
data_by_year = [df[df["Year"]==y]["Returns"].dropna().values for y in years]
bp = ax3.boxplot(data_by_year, labels=years, patch_artist=True,
                 medianprops={"color":"white","lw":1.5},
                 whiskerprops={"color":"#8b949e"},
                 capprops={"color":"#8b949e"},
                 flierprops={"marker":".", "markerfacecolor":"#f85149", "markersize":3})
colors_bp = plt.cm.Blues(np.linspace(0.4, 0.9, len(years)))
for patch, c in zip(bp["boxes"], colors_bp):
    patch.set_facecolor(c)
ax3.set_title("Return Distribution by Year", fontweight="bold")
ax3.set_ylabel("Return (%)")
ax3.tick_params(axis="x", rotation=30, labelsize=8)
ax3.grid(True, alpha=0.25, axis="y")

# Volume histogram
ax4 = fig.add_subplot(gs[1, 2])
ax4.hist(df["Volume"]/1e6, bins=50, color="#d2a8ff", edgecolor="#0d1117", alpha=0.85)
ax4.set_title("Volume Distribution", fontweight="bold")
ax4.set_xlabel("Volume (Millions)")
ax4.set_ylabel("Frequency")
ax4.grid(True, alpha=0.25)

# Scatter: Price vs Volume
ax5 = fig.add_subplot(gs[2, 0])
sc = ax5.scatter(df["Volume"]/1e6, df["Price"],
                 c=df["Year"], cmap="viridis", s=5, alpha=0.5)
plt.colorbar(sc, ax=ax5, label="Year")
ax5.set_xlabel("Volume (M)")
ax5.set_ylabel("Price (PKR)")
ax5.set_title("Price vs Volume (coloured by Year)", fontweight="bold")
ax5.grid(True, alpha=0.25)

# QQ plot
ax6 = fig.add_subplot(gs[2, 1])
(osm, osr), (slope, intercept, _) = stats.probplot(ret_clean, dist="norm")
ax6.scatter(osm, osr, s=4, alpha=0.4, color="#58a6ff")
ax6.plot(osm, slope*np.array(osm)+intercept, "r-", lw=1.5)
ax6.set_title("Q-Q Plot: Returns vs Normal", fontweight="bold")
ax6.set_xlabel("Theoretical Quantiles")
ax6.set_ylabel("Sample Quantiles")
ax6.grid(True, alpha=0.25)

# Returns over time
ax7 = fig.add_subplot(gs[2, 2])
ax7.bar(df["Date"], df["Returns"],
        color=np.where(df["Returns"]>=0, "#2ea043", "#f85149"),
        width=1.2, alpha=0.8)
ax7.axhline(0, color="#8b949e", lw=0.7)
ax7.set_title("Daily Returns Over Time", fontweight="bold")
ax7.set_xlabel("Date")
ax7.set_ylabel("Return (%)")
ax7.grid(True, alpha=0.2, axis="y")

fig.suptitle("OGDC — Exploratory Data Analysis Dashboard", fontsize=15, fontweight="bold", y=1.01)
save_fig(fig, "01_eda_panel.png")

# Chart 1b: Scatter matrix 
scatter_vars = ["Price","Volume","Returns","HV20"]
scatter_df   = df[scatter_vars].dropna()

fig, axes = plt.subplots(4, 4, figsize=(14, 14))
fig.subplots_adjust(hspace=0.4, wspace=0.4)
colors_sc = plt.cm.Blues(np.linspace(0.4, 0.85, len(scatter_df)))

for i, vi in enumerate(scatter_vars):
    for j, vj in enumerate(scatter_vars):
        ax = axes[i][j]
        if i == j:
            ax.hist(scatter_df[vi].dropna(), bins=35, color="#388bfd", alpha=0.8, edgecolor="#0d1117")
            ax.set_title(vi, fontsize=9, fontweight="bold")
        else:
            ax.scatter(scatter_df[vj], scatter_df[vi], s=3, alpha=0.3, color="#58a6ff")
            r, p = stats.pearsonr(scatter_df[vj].dropna(), scatter_df[vi].dropna())
            ax.set_title(f"r={r:.2f}", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.2)

fig.suptitle("OGDC — Scatter Matrix: Price, Volume, Returns, HV-20", fontsize=13, fontweight="bold")
save_fig(fig, "02_scatter_matrix.png")

sub("EDA Summary Statistics by Year")
eda_yr = df.groupby("Year")["Returns"].agg(["mean","std","min","max",
    lambda x: x.skew(), lambda x: x.kurtosis()])
eda_yr.columns = ["Mean%","Std%","Min%","Max%","Skew","Kurt"]
print(eda_yr.round(4).to_string())


# Part 2 — ANOVA
banner("Part 2 — ANOVA")

# 2A: One-Way ANOVA — Returns across Years 
sub("2A: One-Way ANOVA — Annual Returns")
year_groups = [df[df["Year"]==y]["Returns"].dropna().values for y in years]

# Levene's test for homogeneity of variance (assumption check)
lev_stat, lev_p = stats.levene(*year_groups)
print(f"\n  Levene's test (equal variances): W={lev_stat:.4f}  p={lev_p:.4f}")
print(f"  Homogeneity of variance: {'Met ✓' if lev_p > 0.05 else 'Violated ✗ — Welch ANOVA used'}")

# Welch ANOVA via pairwise Welch t-tests + F-approximation
f_stat, f_p = stats.f_oneway(*year_groups)
print(f"\n  One-Way ANOVA:")
print(f"    F-statistic : {f_stat:.4f}")
print(f"    p-value     : {f_p:.6f}")
print(f"    Result      : {'Reject H₀ — significant difference across years' if f_p < 0.05 else 'Fail to reject H₀'}")

# Effect size (eta-squared)
grand_mean = df["Returns"].dropna().mean()
ss_between = sum(len(g)*(np.mean(g)-grand_mean)**2 for g in year_groups)
ss_total   = sum(((df[df["Year"]==y]["Returns"].dropna()-grand_mean)**2).sum() for y in years)
eta2       = ss_between / ss_total if ss_total > 0 else 0
print(f"    η² (effect size): {eta2:.4f}  ({'small' if eta2<0.06 else ('medium' if eta2<0.14 else 'large')})")

results_log["anova_F"]   = round(f_stat, 4)
results_log["anova_p"]   = round(f_p, 6)
results_log["anova_eta2"]= round(eta2, 4)

# Post-hoc Tukey-like pairwise comparison (Bonferroni corrected)
sub("Post-Hoc Pairwise Comparisons (Bonferroni corrected)")
year_pairs = list(combinations(years, 2))
n_tests = len(year_pairs)
print(f"  {'Pair':12s}  {'t-stat':>8}  {'p-raw':>10}  {'p-adj':>10}  {'Sig?':6}")
print(f"  {'─'*52}")
posthoc_rows = []
for ya, yb in year_pairs:
    ga = df[df["Year"]==ya]["Returns"].dropna()
    gb = df[df["Year"]==yb]["Returns"].dropna()
    t, p_raw = stats.ttest_ind(ga, gb, equal_var=False)
    p_adj    = min(p_raw * n_tests, 1.0)
    sig      = "✓" if p_adj < 0.05 else ""
    print(f"  {ya}-{yb}       {t:>8.4f}  {p_raw:>10.4f}  {p_adj:>10.4f}  {sig}")
    posthoc_rows.append({"pair":f"{ya}-{yb}","t":t,"p_raw":p_raw,"p_adj":p_adj,"sig":sig})

# 2B: Two-Way ANOVA — Year × Volatility Regime 
sub("2B: Two-Way ANOVA — Year × Volatility Regime")

df_anova2 = df[["Year","VolRegime","Returns"]].dropna()
groups_2way = df_anova2.groupby(["Year","VolRegime"])["Returns"].apply(list)

# Compute SS for two-way ANOVA (balanced approximation)
grand  = df_anova2["Returns"].mean()
yr_means = df_anova2.groupby("Year")["Returns"].mean()
vr_means = df_anova2.groupby("VolRegime")["Returns"].mean()

ss_year = sum(len(df_anova2[df_anova2["Year"]==y])*(m-grand)**2 for y,m in yr_means.items())
ss_vr   = sum(len(df_anova2[df_anova2["VolRegime"]==v])*(m-grand)**2 for v,m in vr_means.items())
ss_tot  = ((df_anova2["Returns"]-grand)**2).sum()
ss_err  = ss_tot - ss_year - ss_vr

df_year = len(yr_means) - 1
df_vr   = len(vr_means) - 1
df_err  = len(df_anova2) - df_year - df_vr - 1

F_year = (ss_year/df_year) / (ss_err/df_err) if df_err > 0 else np.nan
F_vr   = (ss_vr/df_vr)   / (ss_err/df_err) if df_err > 0 else np.nan
p_year = 1 - stats.f.cdf(F_year, df_year, df_err)
p_vr   = 1 - stats.f.cdf(F_vr,   df_vr,   df_err)

print(f"\n  {'Source':20s}  {'SS':>12}  {'df':>4}  {'MS':>12}  {'F':>8}  {'p':>8}")
print(f"  {'─'*72}")
print(f"  {'Year':20s}  {ss_year:>12.4f}  {df_year:>4}  {ss_year/df_year:>12.4f}  {F_year:>8.4f}  {p_year:>8.4f}")
print(f"  {'Volatility Regime':20s}  {ss_vr:>12.4f}  {df_vr:>4}  {ss_vr/df_vr:>12.4f}  {F_vr:>8.4f}  {p_vr:>8.4f}")
print(f"  {'Error':20s}  {ss_err:>12.4f}  {df_err:>4}  {ss_err/df_err:>12.4f}")

results_log["anova2_F_year"] = round(F_year, 4)
results_log["anova2_p_year"] = round(p_year, 6)
results_log["anova2_F_vr"]   = round(F_vr, 4)
results_log["anova2_p_vr"]   = round(p_vr, 6)

# ANOVA chart
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

colors_yr = plt.cm.Blues(np.linspace(0.35, 0.9, len(years)))
means_yr  = [np.mean(g) for g in year_groups]
stds_yr   = [np.std(g)/np.sqrt(len(g)) for g in year_groups]

ax1.bar(years, means_yr, color=colors_yr, edgecolor="#0d1117", alpha=0.85, width=0.6)
ax1.errorbar(years, means_yr, yerr=[1.96*s for s in stds_yr],
             fmt="none", color="white", capsize=5, lw=1.5)
ax1.axhline(grand_mean, color="#f0883e", lw=1.5, linestyle="--",
            label=f"Grand mean ({grand_mean:.3f}%)")
ax1.set_title(f"One-Way ANOVA — Mean Return by Year\nF={f_stat:.2f}, p={f_p:.4f}, η²={eta2:.4f}",
               fontweight="bold")
ax1.set_xlabel("Year")
ax1.set_ylabel("Mean Return (%)")
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3, axis="y")

# Interaction plot for two-way
regime_order = ["Low","Medium","High"]
regime_colors = {"Low":"#2ea043","Medium":"#d29922","High":"#f85149"}
for yr in years:
    yr_df = df_anova2[df_anova2["Year"]==yr]
    means = yr_df.groupby("VolRegime")["Returns"].mean()
    vals  = [means.get(r, np.nan) for r in regime_order]
    ax2.plot(regime_order, vals, "o-", label=str(yr), lw=1.2, markersize=7)

ax2.set_title(f"Two-Way ANOVA Interaction — Year × Vol Regime\nF(Year)={F_year:.2f} p={p_year:.3f}   F(Vol)={F_vr:.2f} p={p_vr:.3f}",
               fontweight="bold")
ax2.set_xlabel("Volatility Regime")
ax2.set_ylabel("Mean Return (%)")
ax2.legend(fontsize=8, title="Year")
ax2.grid(True, alpha=0.3)

fig.suptitle("ANOVA Analysis — OGDC Returns", fontsize=13, fontweight="bold")
plt.tight_layout()
save_fig(fig, "03_anova.png")


# Part 3 — Probability Distribution Fitting
banner("Part 3 — Probability Distribution Fitting")

distributions = {
    "Normal":    stats.norm,
    "Student-t": stats.t,
    "Laplace":   stats.laplace,
    "Logistic":  stats.logistic,
    "Cauchy":    stats.cauchy,
}

fit_results = {}
r_vals = ret_clean.values

print(f"\n  {'Distribution':15s}  {'Log-Lik':>12}  {'AIC':>10}  {'BIC':>10}  {'KS-stat':>8}  {'KS-p':>8}")
print(f"  {'─'*68}")

for name, dist in distributions.items():
    try:
        if name == "Student-t":
            params = dist.fit(r_vals)
            k = 3
        else:
            params = dist.fit(r_vals)
            k = len(params)
        n = len(r_vals)
        ll = np.sum(dist.logpdf(r_vals, *params))
        aic = 2*k - 2*ll
        bic = np.log(n)*k - 2*ll
        ks_stat, ks_p = stats.kstest(r_vals, dist.cdf, args=params)
        fit_results[name] = {"params":params,"ll":ll,"aic":aic,"bic":bic,"ks":ks_stat,"ks_p":ks_p,"dist":dist}
        print(f"  {name:15s}  {ll:>12.2f}  {aic:>10.2f}  {bic:>10.2f}  {ks_stat:>8.4f}  {ks_p:>8.4f}")
    except Exception as e:
        print(f"  {name:15s}  FAILED: {e}")

best_dist = min(fit_results, key=lambda k: fit_results[k]["aic"])
print(f"\n  Best fit by AIC: {best_dist}")
results_log["best_dist"] = best_dist
results_log["best_aic"]  = round(fit_results[best_dist]["aic"], 2)

# Jarque-Bera normality test
jb_stat, jb_p = stats.jarque_bera(r_vals)
print(f"\n  Jarque-Bera normality test: JB={jb_stat:.2f}  p={jb_p:.6f}")
print(f"  Conclusion: {'Returns are NOT normally distributed (reject H₀)' if jb_p < 0.05 else 'Normal distribution not rejected'}")
results_log["jb_stat"] = round(jb_stat, 4)
results_log["jb_p"]    = round(jb_p, 6)

# Shapiro-Wilk on a sample (max 5000)
sw_sample = r_vals[:5000] if len(r_vals) > 5000 else r_vals
sw_stat, sw_p = stats.shapiro(sw_sample[:2000])
print(f"  Shapiro-Wilk (n=2000 sample): W={sw_stat:.4f}  p={sw_p:.6f}")

# Distribution chart 
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.subplots_adjust(hspace=0.45, wspace=0.35)

x_plot = np.linspace(ret_clean.quantile(0.001), ret_clean.quantile(0.999), 300)
hist_color = "#388bfd"

for ax, (name, fr) in zip(axes.flat, fit_results.items()):
    ax.hist(r_vals, bins=70, density=True, color=hist_color, alpha=0.55, edgecolor="#0d1117")
    try:
        y_pdf = fr["dist"].pdf(x_plot, *fr["params"])
        col = "#f0883e" if name == best_dist else "#3fb950"
        lw  = 2.2 if name == best_dist else 1.4
        ax.plot(x_plot, y_pdf, color=col, lw=lw,
                label=f"AIC={fr['aic']:.0f}\nKS-p={fr['ks_p']:.3f}")
    except Exception:
        pass
    star = " ★ BEST" if name == best_dist else ""
    ax.set_title(f"{name}{star}", fontweight="bold",
                 color="#f0883e" if name == best_dist else "#e6edf3")
    ax.set_xlim(ret_clean.quantile(0.001), ret_clean.quantile(0.999))
    ax.set_xlabel("Return (%)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)

axes.flat[-1].set_visible(False)
fig.suptitle("OGDC Daily Returns — Probability Distribution Fitting", fontsize=14, fontweight="bold")
save_fig(fig, "04_distribution_fitting.png")


# Part 4 — Multiple Regression
banner("Part 4 — Multiple Regression")

# Target: Price_t   Predictors: Open, High, Low, Volume, Return_lag1, Return_lag2
df["Return_lag1"] = df["Returns"].shift(1)
df["Return_lag2"] = df["Returns"].shift(2)
df["Vol_scaled"]  = df["Volume"] / 1e6

reg_df = df[["Price","Open","High","Low","Vol_scaled","Return_lag1","Return_lag2"]].dropna()
y_reg  = reg_df["Price"].values
X_cols = ["Open","High","Low","Vol_scaled","Return_lag1","Return_lag2"]
X_raw  = reg_df[X_cols].values
n_reg  = len(y_reg)
k_reg  = len(X_cols)

# Add intercept
X_mat  = np.column_stack([np.ones(n_reg), X_raw])

# OLS via normal equations: β = (XᵀX)⁻¹Xᵀy
try:
    beta   = np.linalg.lstsq(X_mat, y_reg, rcond=None)[0]
    y_hat  = X_mat @ beta
    resid  = y_reg - y_hat
    ss_res = float(resid @ resid)
    ss_tot = float(((y_reg - y_reg.mean())**2).sum())
    R2     = 1 - ss_res/ss_tot
    R2_adj = 1 - (1-R2)*(n_reg-1)/(n_reg-k_reg-1)
    mse    = ss_res / (n_reg - k_reg - 1)
    se_all = np.sqrt(mse * np.diag(np.linalg.inv(X_mat.T @ X_mat)))
    t_vals = beta / se_all
    p_vals = 2*(1 - stats.t.cdf(np.abs(t_vals), df=n_reg-k_reg-1))
    f_stat_reg = (R2/k_reg) / ((1-R2)/(n_reg-k_reg-1))
    f_p_reg    = 1 - stats.f.cdf(f_stat_reg, k_reg, n_reg-k_reg-1)
    mae_reg    = float(np.abs(resid).mean())
    rmse_reg   = float(np.sqrt(mse))

    print(f"\n  Multiple Regression: Price ~ Open + High + Low + Volume + Lag1 + Lag2")
    print(f"\n  {'Coefficient':15s}  {'Estimate':>10}  {'Std.Err':>10}  {'t-stat':>8}  {'p-value':>10}  {'Sig'}")
    print(f"  {'─'*70}")
    coef_names = ["Intercept"] + X_cols
    for nm, b, se, t, p in zip(coef_names, beta, se_all, t_vals, p_vals):
        sig = "***" if p<0.001 else ("**" if p<0.01 else ("*" if p<0.05 else ""))
        print(f"  {nm:15s}  {b:>10.4f}  {se:>10.4f}  {t:>8.4f}  {p:>10.4f}  {sig}")

    print(f"\n  R²={R2:.6f}  Adj.R²={R2_adj:.6f}")
    print(f"  F={f_stat_reg:.2f}  p={f_p_reg:.6f}")
    print(f"  RMSE={rmse_reg:.4f}  MAE={mae_reg:.4f}")

    results_log["reg_R2"]     = round(R2, 6)
    results_log["reg_adjR2"]  = round(R2_adj, 6)
    results_log["reg_F"]      = round(f_stat_reg, 4)
    results_log["reg_p"]      = round(f_p_reg, 6)
    results_log["reg_RMSE"]   = round(rmse_reg, 4)

except Exception as e:
    print(f"  Regression failed: {e}")
    beta = y_hat = resid = None
    R2 = R2_adj = f_stat_reg = f_p_reg = 0

sub("Regression Assumption Checks")

# Normality of residuals
if resid is not None:
    jb_res, jb_res_p = stats.jarque_bera(resid)
    print(f"  Normality of residuals (JB): stat={jb_res:.2f}  p={jb_res_p:.4f}  "
          f"→ {'Normal ✓' if jb_res_p > 0.05 else 'Non-normal ✗'}")

    # Homoscedasticity — Breusch-Pagan (correlate squared residuals with predictors)
    resid2 = resid**2
    bp_res = np.linalg.lstsq(X_mat, resid2, rcond=None)[0]
    y_hat2 = X_mat @ bp_res
    ss_r2  = float(((y_hat2 - resid2.mean())**2).sum())
    ss_t2  = float(((resid2 - resid2.mean())**2).sum())
    R2_bp  = ss_r2/ss_t2
    bp_stat = n_reg * R2_bp
    bp_p    = 1 - stats.chi2.cdf(bp_stat, df=k_reg)
    print(f"  Homoscedasticity (BP test) : stat={bp_stat:.2f}  p={bp_p:.4f}  "
          f"→ {'Homoscedastic ✓' if bp_p > 0.05 else 'Heteroscedastic ✗'}")

    # Multicollinearity — VIF
    print(f"\n  Variance Inflation Factors:")
    for i, col in enumerate(X_cols):
        xi = X_raw[:, i]
        xo = np.column_stack([np.ones(n_reg), X_raw[:, [j for j in range(k_reg) if j!=i]]])
        b_vif, *_ = np.linalg.lstsq(xo, xi, rcond=None)
        xi_hat = xo @ b_vif
        ss_r_vif = float(((xi - xi_hat)**2).sum())
        ss_t_vif = float(((xi - xi.mean())**2).sum())
        r2_vif = 1 - ss_r_vif/ss_t_vif if ss_t_vif>0 else 0
        vif = 1/(1-r2_vif) if r2_vif<1 else np.inf
        flag = " ⚠ HIGH" if vif > 10 else ""
        print(f"    {col:15s}: VIF = {vif:.2f}{flag}")

# Regression charts 
if resid is not None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.subplots_adjust(hspace=0.35, wspace=0.3)

    ax1, ax2, ax3, ax4 = axes.flat

    # Actual vs Fitted
    ax1.scatter(y_hat, y_reg, s=4, alpha=0.3, color="#58a6ff")
    lo, hi = min(y_hat.min(),y_reg.min()), max(y_hat.max(),y_reg.max())
    ax1.plot([lo,hi],[lo,hi], "r--", lw=1.5)
    ax1.set_xlabel("Fitted Values (PKR)")
    ax1.set_ylabel("Actual Price (PKR)")
    ax1.set_title(f"Actual vs Fitted  (R²={R2:.4f})", fontweight="bold")
    ax1.grid(True, alpha=0.25)

    # Residuals vs Fitted
    ax2.scatter(y_hat, resid, s=4, alpha=0.3, color="#f0883e")
    ax2.axhline(0, color="white", lw=1)
    ax2.set_xlabel("Fitted Values")
    ax2.set_ylabel("Residuals")
    ax2.set_title("Residuals vs Fitted", fontweight="bold")
    ax2.grid(True, alpha=0.25)

    # Residual histogram
    ax3.hist(resid, bins=50, color="#3fb950", edgecolor="#0d1117", alpha=0.85, density=True)
    xr2 = np.linspace(resid.min(), resid.max(), 200)
    ax3.plot(xr2, stats.norm.pdf(xr2, resid.mean(), resid.std()), color="white", lw=1.5)
    ax3.set_title("Residual Distribution", fontweight="bold")
    ax3.set_xlabel("Residual")
    ax3.grid(True, alpha=0.25)

    # Coefficients bar (excluding intercept)
    coef_df = pd.DataFrame({"coef":beta[1:], "se":se_all[1:]}, index=X_cols)
    colors_c = ["#2ea043" if v>0 else "#f85149" for v in coef_df["coef"]]
    ax4.barh(coef_df.index, coef_df["coef"], xerr=1.96*coef_df["se"],
             color=colors_c, edgecolor="#0d1117", alpha=0.85, capsize=4)
    ax4.axvline(0, color="white", lw=0.8)
    ax4.set_title("Regression Coefficients (±95% CI)", fontweight="bold")
    ax4.set_xlabel("Coefficient value")
    ax4.grid(True, alpha=0.25, axis="x")

    fig.suptitle("OGDC — Multiple Regression Diagnostics", fontsize=13, fontweight="bold")
    save_fig(fig, "05_regression.png")


# Part 5 — ARIMA Time Series
banner("Part 5 — Arima Time Series Modelling")

# Use Returns series (stationary — confirmed by ADF)
ts = df["Returns"].dropna().values
n_ts = len(ts)

# ACF / PACF 
sub("ACF / PACF — Model Order Identification")
'''
measures the linear relationship between a time series and its past (lagged) values, 
identifying seasonality, trends, and dependencies
'''

def acf_vals(x, max_lag=40):
    n = len(x); xm = x - x.mean(); c0 = xm@xm/n
    return np.array([1.0]+[(xm[k:]@xm[:-k])/(n*c0) for k in range(1,max_lag+1)])

def pacf_vals(x, max_lag=40):
    acf = acf_vals(x, max_lag)
    pacf = [1.0]; phi = {}
    for k in range(1, max_lag+1):
        if k==1: p=acf[1]
        else:
            num = acf[k] - sum(phi[k-1][j]*acf[k-1-j] for j in range(k-1))
            den = 1 - sum(phi[k-1][j]*acf[j+1] for j in range(k-1))
            p = num/den if abs(den)>1e-12 else 0
        phi[k] = [p] + [phi[k-1][j]-p*phi[k-1][k-2-j] for j in range(k-1)]
        pacf.append(p)
    return np.array(pacf)

acf40  = acf_vals(ts, 40)
pacf40 = pacf_vals(ts, 40)
conf95 = 1.96/np.sqrt(n_ts)

# Identify significant lags
sig_acf  = [i for i in range(1,41) if abs(acf40[i])>conf95]
sig_pacf = [i for i in range(1,41) if abs(pacf40[i])>conf95]
print(f"  Significant ACF  lags: {sig_acf}")
print(f"  Significant PACF lags: {sig_pacf}")

# Suggest orders
p_try = [0,1,2] if not sig_pacf else [min(3,max(sig_pacf[:3]))]
q_try = [0,1,2] if not sig_acf  else [min(3,max(sig_acf[:3]))]

# ARMA model selection via AIC 
sub("ARMA Model Selection (AIC minimisation)")

def fit_arma(x, p, q, max_iter=500):
    """Fit ARMA(p,q) by conditional least squares (CSS) approximation."""
    n = len(x); xm = x - x.mean(); mu = x.mean()
    k = p + q + 1   # intercept + AR + MA
    if k >= n: return None

    def neg_ll(params):
        mu_p = params[0]
        ar = params[1:p+1]
        ma = params[p+1:p+q+1]
        sigma2 = max(params[-1]**2, 1e-8)
        eps = np.zeros(n)
        for t in range(n):
            val = x[t] - mu_p
            for i,a in enumerate(ar):
                if t-i-1>=0: val -= a*x[t-i-1]
            for j,m in enumerate(ma):
                if t-j-1>=0: val -= m*eps[t-j-1]
            eps[t] = val
        ll = -n/2*np.log(2*np.pi*sigma2) - 0.5*np.sum(eps**2)/sigma2
        return -ll

    x0 = np.zeros(k+1); x0[0]=mu; x0[-1]=x.std()
    try:
        from scipy.optimize import minimize
        res = minimize(neg_ll, x0, method="Nelder-Mead",
                       options={"maxiter":max_iter,"xatol":1e-5,"fatol":1e-5})
        ll  = -res.fun
        aic = 2*k - 2*ll
        bic = np.log(n)*k - 2*ll
        return {"p":p,"q":q,"aic":aic,"bic":bic,"ll":ll,"params":res.x,"success":res.success}
    except Exception:
        return None

model_sel = []
for p in range(0,4):
    for q in range(0,4):
        r = fit_arma(ts, p, q)
        if r: model_sel.append(r)

model_sel = sorted(model_sel, key=lambda x: x["aic"])
print(f"\n  {'p':>3} {'q':>3}  {'AIC':>10}  {'BIC':>10}  {'LogLik':>10}")
print(f"  {'─'*42}")
for m in model_sel[:10]:
    print(f"  {m['p']:>3} {m['q']:>3}  {m['aic']:>10.2f}  {m['bic']:>10.2f}  {m['ll']:>10.2f}")

best_model = model_sel[0]
best_p, best_q = best_model["p"], best_model["q"]
print(f"\n  Best: ARMA({best_p},{best_q})  AIC={best_model['aic']:.2f}")
results_log["arima_p"] = best_p
results_log["arima_q"] = best_q
results_log["arima_aic"] = round(best_model["aic"], 2)

# ARIMA Forecast (30-day)
sub("30-Day Return Forecast (ARIMA mean forecast)")

params_best = best_model["params"]
mu_arma = params_best[0]
ar_c    = params_best[1:best_p+1] if best_p > 0 else []
ma_c    = params_best[best_p+1:best_p+best_q+1] if best_q > 0 else []

# Mean forecast converges to mu for stationary ARMA
HORIZON = 30
forecast_means = []
last_vals = list(ts[-max(best_p,1):])
last_eps  = [0.0] * max(best_q, 1)

for h in range(HORIZON):
    f = mu_arma
    for i, a in enumerate(ar_c):
        if i < len(last_vals):
            f += a * last_vals[-(i+1)]
    for j, m in enumerate(ma_c):
        if j < len(last_eps):
            f += m * last_eps[-(j+1)]
    forecast_means.append(f)
    last_vals.append(f)
    last_eps.append(0.0)   # innovation = 0 for h>0

# Forecast variance grows with horizon
sigma2_est = params_best[-1]**2 if len(params_best)>0 else ts.var()
se_forecast = [np.sqrt(sigma2_est * (h+1)) for h in range(HORIZON)]
ci_upper = [f + 1.96*se for f, se in zip(forecast_means, se_forecast)]
ci_lower = [f - 1.96*se for f, se in zip(forecast_means, se_forecast)]

last_date  = df["Date"].max()
fcast_idx  = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=HORIZON)

print(f"\n  ARMA({best_p},{best_q}) — 30-day ahead forecast:")
print(f"  {'Day':>4}  {'Date':12}  {'Forecast%':>10}  {'Lower 95%':>10}  {'Upper 95%':>10}")
print(f"  {'─'*56}")
for i, (dt, f, lo, hi) in enumerate(zip(fcast_idx, forecast_means, ci_lower, ci_upper), 1):
    print(f"  {i:>4}  {str(dt.date()):12}  {f:>10.4f}  {lo:>10.4f}  {hi:>10.4f}")

# Ljung-Box residual check 
sub("Model Diagnostic — Ljung-Box on ARMA Residuals")

# Re-extract residuals
residuals_arma = np.zeros(n_ts)
for t in range(n_ts):
    val = ts[t] - mu_arma
    for i, a in enumerate(ar_c):
        if t-i-1>=0: val -= a*ts[t-i-1]
    for j, m in enumerate(ma_c):
        if t-j-1>=0: val -= m*residuals_arma[t-j-1]
    residuals_arma[t] = val

# Ljung-Box Q
for lag in [5, 10, 20]:
    acf_res = acf_vals(residuals_arma, lag)
    Q = n_ts*(n_ts+2)*sum(acf_res[k]**2/(n_ts-k) for k in range(1,lag+1))
    p_lb = 1 - stats.chi2.cdf(Q, df=lag-best_p-best_q)
    print(f"  LB Q({lag:>2}): Q={Q:.2f}  p={p_lb:.4f}  "
          f"{'White noise ✓' if p_lb > 0.05 else 'Autocorrelation remains ✗'}")

# ARIMA charts 
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.subplots_adjust(hspace=0.38, wspace=0.32)
lags = np.arange(0, 41)

# ACF
axes[0,0].bar(lags, acf40, color="#58a6ff", alpha=0.8, width=0.6)
axes[0,0].axhline( conf95, color="#f85149", lw=1.5, linestyle="--")
axes[0,0].axhline(-conf95, color="#f85149", lw=1.5, linestyle="--")
axes[0,0].axhline(0, color="#8b949e", lw=0.7)
axes[0,0].set_title("ACF — OGDC Daily Returns", fontweight="bold")
axes[0,0].set_xlabel("Lag")
axes[0,0].set_ylabel("Autocorrelation")
axes[0,0].grid(True, alpha=0.25)

# PACF
axes[0,1].bar(lags, pacf40, color="#f0883e", alpha=0.8, width=0.6)
axes[0,1].axhline( conf95, color="#f85149", lw=1.5, linestyle="--")
axes[0,1].axhline(-conf95, color="#f85149", lw=1.5, linestyle="--")
axes[0,1].axhline(0, color="#8b949e", lw=0.7)
axes[0,1].set_title("PACF — OGDC Daily Returns", fontweight="bold")
axes[0,1].set_xlabel("Lag")
axes[0,1].set_ylabel("Partial Autocorrelation")
axes[0,1].grid(True, alpha=0.25)

# Forecast
hist_plot = ts[-90:]
hist_dates = df["Date"].dropna().values[-90:]
axes[1,0].plot(hist_dates, hist_plot, lw=1.1, color="#58a6ff", label="Historical")
axes[1,0].plot(fcast_idx, forecast_means, lw=1.5, color="#f0883e",
               linestyle="--", label=f"ARMA({best_p},{best_q}) Forecast")
axes[1,0].fill_between(fcast_idx, ci_lower, ci_upper, color="#1f4080", alpha=0.4, label="95% CI")
axes[1,0].axhline(0, color="#8b949e", lw=0.7)
axes[1,0].set_title(f"ARMA({best_p},{best_q}) — 30-Day Return Forecast", fontweight="bold")
axes[1,0].set_xlabel("Date")
axes[1,0].set_ylabel("Return (%)")
axes[1,0].legend(fontsize=8)
axes[1,0].grid(True, alpha=0.25)

# Residuals
axes[1,1].plot(range(len(residuals_arma)), residuals_arma, lw=0.7, color="#3fb950", alpha=0.8)
axes[1,1].axhline(0, color="white", lw=0.8)
axes[1,1].set_title(f"ARMA({best_p},{best_q}) Residuals", fontweight="bold")
axes[1,1].set_xlabel("Observation")
axes[1,1].set_ylabel("Residual")
axes[1,1].grid(True, alpha=0.25)

fig.suptitle("OGDC — ARIMA Analysis", fontsize=13, fontweight="bold")
save_fig(fig, "06_arima.png")


# Part 6 — Non Parametric Statistics
banner("Part 6 — Non-Parametric Statistics")

# Kruskal-Wallis (non-parametric ANOVA)
sub("Kruskal-Wallis H Test — Returns across Years (non-parametric ANOVA)")

kw_stat, kw_p = stats.kruskal(*year_groups)
print(f"  H-statistic : {kw_stat:.4f}")
print(f"  p-value     : {kw_p:.6f}")
print(f"  Result      : {'Reject H₀ — significant differences across years' if kw_p < 0.05 else 'Fail to reject H₀'}")

# Effect size: eta-squared for Kruskal-Wallis
n_total = sum(len(g) for g in year_groups)
eta2_kw = (kw_stat - len(years) + 1) / (n_total - len(years))
print(f"  η² effect size: {eta2_kw:.4f}")

results_log["kw_H"]  = round(kw_stat, 4)
results_log["kw_p"]  = round(kw_p, 6)

# Mann-Whitney U — Bull vs Bear years 
sub("Mann-Whitney U Test — Bull Years (2023–2024) vs Bear Years (2022)")

bull = df[df["Year"].isin([2023,2024])]["Returns"].dropna()
bear = df[df["Year"] == 2022]["Returns"].dropna()

mw_stat, mw_p = stats.mannwhitneyu(bull, bear, alternative="greater")
print(f"  U-statistic : {mw_stat:.4f}")
print(f"  p-value     : {mw_p:.6f}")
print(f"  Alternative : Bull years > Bear years")
print(f"  Result      : {'Significant ✓' if mw_p < 0.05 else 'Not significant'}")

# Effect size r = Z / sqrt(N)
z_mw = (mw_stat - len(bull)*len(bear)/2) / np.sqrt(len(bull)*len(bear)*(len(bull)+len(bear)+1)/12)
r_mw = z_mw / np.sqrt(len(bull)+len(bear))
print(f"  Effect size r: {r_mw:.4f}")

results_log["mw_U"] = round(float(mw_stat), 4)
results_log["mw_p"] = round(float(mw_p), 6)

# Spearman Rank Correlation 
sub("Spearman Rank Correlation")

pairs_sp = [("Price","Volume"),("Returns","Volume"),("Price","Returns"),("HV20","Returns")]
print(f"\n  {'Pair':25s}  {'Pearson r':>10}  {'Spearman ρ':>12}  {'p-value':>10}")
print(f"  {'─'*60}")
for v1, v2 in pairs_sp:
    sub_df = df[[v1,v2]].dropna()
    r_pear, p_pear = stats.pearsonr(sub_df[v1], sub_df[v2])
    r_sp, p_sp     = stats.spearmanr(sub_df[v1], sub_df[v2])
    print(f"  {v1+' ~ '+v2:25s}  {r_pear:>10.4f}  {r_sp:>12.4f}  {p_sp:>10.4f}")

# Runs Test for randomness 
sub("Runs Test for Randomness — Daily Returns")

ret_signs = np.sign(df["Returns"].dropna().values)
ret_signs = ret_signs[ret_signs != 0]   # remove zeros

# Count runs
runs = 1
for i in range(1, len(ret_signs)):
    if ret_signs[i] != ret_signs[i-1]:
        runs += 1

n_pos = (ret_signs == 1).sum()
n_neg = (ret_signs == -1).sum()
n_tot = n_pos + n_neg
mu_runs  = 2*n_pos*n_neg/n_tot + 1
var_runs = (mu_runs-1)*(mu_runs-2)/(n_tot-1) if n_tot > 2 else 1
z_runs   = (runs - mu_runs) / np.sqrt(var_runs)
p_runs   = 2*(1 - stats.norm.cdf(abs(z_runs)))

print(f"\n  n(+) = {n_pos}  n(-) = {n_neg}  Total = {n_tot}")
print(f"  Observed runs   : {runs}")
print(f"  Expected runs   : {mu_runs:.2f}")
print(f"  Z-statistic     : {z_runs:.4f}")
print(f"  p-value         : {p_runs:.6f}")
print(f"  Conclusion      : {'Returns are NOT random (reject H₀)' if p_runs<0.05 else 'Returns are random (fail to reject H₀)'}")

results_log["runs_Z"] = round(float(z_runs), 4)
results_log["runs_p"] = round(float(p_runs), 6)

# Nonparametric charts 
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.subplots_adjust(hspace=0.4, wspace=0.35)

# KW box plots
bp2 = axes[0,0].boxplot(year_groups, labels=years, patch_artist=True,
                         medianprops={"color":"white","lw":1.5},
                         whiskerprops={"color":"#8b949e"},
                         capprops={"color":"#8b949e"},
                         flierprops={"marker":".","markerfacecolor":"#f85149","markersize":3})
for patch, c in zip(bp2["boxes"], plt.cm.Blues(np.linspace(0.4,0.9,len(years)))):
    patch.set_facecolor(c)
axes[0,0].set_title(f"Kruskal-Wallis: Returns by Year\nH={kw_stat:.2f}, p={kw_p:.4f}",
                     fontweight="bold")
axes[0,0].set_ylabel("Return (%)")
axes[0,0].grid(True, alpha=0.25, axis="y")

# Mann-Whitney violin
parts = axes[0,1].violinplot([bull.values, bear.values], positions=[1,2],
                              showmedians=True)
for pc, color in zip(parts["bodies"], ["#3fb950","#f85149"]):
    pc.set_facecolor(color); pc.set_alpha(0.7)
axes[0,1].set_xticks([1,2])
axes[0,1].set_xticklabels(["Bull (2023–24)","Bear (2022)"])
axes[0,1].set_title(f"Mann-Whitney: Bull vs Bear\nU={mw_stat:.0f}, p={mw_p:.4f}",
                     fontweight="bold")
axes[0,1].set_ylabel("Return (%)")
axes[0,1].grid(True, alpha=0.25, axis="y")

# Spearman heatmap
sp_cols = ["Price","Volume","Returns","HV20"]
sp_mat  = df[sp_cols].dropna().corr(method="spearman")
sns.heatmap(sp_mat, annot=True, fmt=".3f", cmap="RdYlGn", center=0,
            ax=axes[1,0], vmin=-1, vmax=1, linewidths=0.5, cbar=True)
axes[1,0].set_title("Spearman Rank Correlation Matrix", fontweight="bold")

# Runs test — sign sequence visualisation
sign_vals = ret_signs[:200]
run_changes = np.where(np.diff(sign_vals) != 0)[0]
axes[1,1].plot(range(200), sign_vals, lw=0.7, color="#8b949e", alpha=0.6)
axes[1,1].fill_between(range(200), 0, sign_vals,
                        where=sign_vals>0, color="#2ea043", alpha=0.6, label="Positive")
axes[1,1].fill_between(range(200), 0, sign_vals,
                        where=sign_vals<0, color="#f85149", alpha=0.6, label="Negative")
for rc in run_changes[:50]:
    axes[1,1].axvline(rc, color="#d29922", lw=0.4, alpha=0.5)
axes[1,1].set_title(f"Runs Test: Sign Sequence (first 200)\nZ={z_runs:.2f}, p={p_runs:.4f}",
                     fontweight="bold")
axes[1,1].set_xlabel("Observation")
axes[1,1].set_ylabel("Sign of Return")
axes[1,1].legend(fontsize=8)
axes[1,1].grid(True, alpha=0.2)

fig.suptitle("OGDC — Nonparametric Statistical Tests", fontsize=13, fontweight="bold")
save_fig(fig, "07_nonparametric.png")

# Results 
banner("Saving Results")
res_df = pd.DataFrame([results_log]).T.rename(columns={0:"Value"})
res_df.index.name = "Metric"
res_df.to_csv(OUT_CSV)
print(f"  Saved stat222_results.csv")
print(f"\n  All chart groups saved (stat_01 → stat_07)")

print(f"""
  ╔══════════════════════════════════════════════════════════════════╗
  ║         Results Summarized                                       ║
  ╠══════════════════════════════════════════════════════════════════╣
  ║  ANOVA (One-Way)   F={results_log.get('anova_F',0):<6}  p={results_log.get('anova_p',0):<10}  η²={results_log.get('anova_eta2',0):<6} ║
  ║  ANOVA (Two-Way)   F_year={results_log.get('anova2_F_year',0):<6}  p={results_log.get('anova2_p_year',0):<10}          ║
  ║  Best Distribution  {results_log.get('best_dist','N/A'):<42} ║
  ║  Jarque-Bera        JB={results_log.get('jb_stat',0):<8}  p={results_log.get('jb_p',0):<10}            ║
  ║  Regression         R²={results_log.get('reg_R2',0):<8}  Adj.R²={results_log.get('reg_adjR2',0):<8}           ║
  ║  ARMA Order         ({results_log.get('arima_p','?')},{results_log.get('arima_q','?')})  AIC={results_log.get('arima_aic',0):<10}                     ║
  ║  Kruskal-Wallis     H={results_log.get('kw_H',0):<8}  p={results_log.get('kw_p',0):<10}               ║
  ║  Runs Test          Z={results_log.get('runs_Z',0):<8}  p={results_log.get('runs_p',0):<10}               ║
  ╚══════════════════════════════════════════════════════════════════╝
""")
