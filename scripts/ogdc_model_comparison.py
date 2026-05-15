"""
ogdc_model_comparison.py
========================
Standalone comparison script for all 4 OGDC stock prediction models.
Retrains each model from ogdc_with_regimes.csv, evaluates on the
chronological test set (last 20%), generates a full markdown report,
and saves all comparison visualisations as PNG files.

Dependencies: pandas, numpy, scikit-learn, scipy, matplotlib, seaborn
Run: python ogdc_model_comparison.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import warnings
warnings.filterwarnings("ignore")

import os
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    mean_absolute_error, mean_squared_error, r2_score, silhouette_score,
    precision_recall_fscore_support,
)

# hardcoded paths
base = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base, "..", "data", "processed")
img_dir = os.path.join(base, "..", "outputs", "images")

os.makedirs(data_dir, exist_ok=True)
os.makedirs(img_dir, exist_ok=True)

data_path = os.path.join(data_dir, "ogdc_with_regimes.csv")
report_path = os.path.join(data_dir, "model_comparison_report.md")
img_prefix = os.path.join(img_dir, "comparison_")

random_state = 42

feature_cols = [
    "Return_lag1", "Return_lag2", "Return_lag3", "Return_lag5",
    "Return_roll_mean5", "Return_roll_std5",
    "Return_roll_mean10", "Return_roll_std10",
    "RSI", "MACD", "MACD_signal", "MACD_hist",
    "DayOfWeek", "Month", "Quarter",
    "Open", "High", "Low", "Volume", "ChangeP",
]

# helpers
_report_lines = []

def rpt(*lines):
    for line in lines:
        _report_lines.append(line)
        print(line)

def save_fig(fig, name):
    path = img_prefix + name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return "comparison_" + name

def section(title, level=2):
    bar = "=" * 68
    print(f"\n{bar}\n  {title}\n{bar}")

def mape(actual, predicted):
    mask = actual != 0
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)

def bootstrap_auc(y_true, y_score, n_boot=500, ci=0.95, seed=42):
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) < 2:
            continue
        try:
            aucs.append(roc_auc_score(y_true[idx], y_score[idx]))
        except Exception:
            pass
    lo = np.percentile(aucs, (1 - ci) / 2 * 100)
    hi = np.percentile(aucs, (1 + ci) / 2 * 100)
    return float(np.mean(aucs)), lo, hi

def mcnemar_test(y_true, pred1, pred2):
    b = np.sum((pred1 == y_true) & (pred2 != y_true))
    c = np.sum((pred1 != y_true) & (pred2 == y_true))
    n = b + c
    if n == 0:
        return 1.0, 0.0
    chi2 = (abs(b - c) - 1) ** 2 / n
    pval = 1 - stats.chi2.cdf(chi2, df=1)
    return chi2, pval

def diebold_mariano(actual, pred1, pred2, h=1):
    e1 = actual - pred1
    e2 = actual - pred2
    d = e1 ** 2 - e2 ** 2
    n = len(d)
    d_mean = d.mean()
    gamma0 = np.var(d, ddof=1)
    nw_var = gamma0
    for k in range(1, h):
        gamma_k = np.cov(d[k:], d[:-k])[0, 1]
        nw_var += 2 * (1 - k / h) * gamma_k
    dm = d_mean / np.sqrt(max(nw_var / n, 1e-12))
    pval = 2 * (1 - stats.norm.cdf(abs(dm)))
    return float(dm), float(pval)

# step 0 - load data
section("step 0 - loading data")

df = pd.read_csv(data_path, index_col="Date", parse_dates=True)
df.sort_index(inplace=True)
df.dropna(subset=["Returns", "Direction"], inplace=True)

feature_cols = [c for c in feature_cols if c in df.columns]

split_idx = int(len(df) * 0.80)
train = df.iloc[:split_idx]
test = df.iloc[split_idx:]

x_train = train[feature_cols]
x_test = test[feature_cols]
y_cls_train = train["Direction"].astype(int)
y_cls_test = test["Direction"].astype(int)
y_reg_train = train["Returns"]
y_reg_test = test["Returns"]

test_start = test.index.min().strftime("%Y-%m-%d")
test_end = test.index.max().strftime("%Y-%m-%d")
n_test = len(test)

print(f"train: {len(train)} rows | test: {n_test} rows ({test_start} to {test_end})")

# step 1 - train all 4 models
section("step 1 - training all models")
tscv3 = TimeSeriesSplit(n_splits=3)
tscv5 = TimeSeriesSplit(n_splits=5)

# model 1: random forest
print("training random forest classifier...")
rf = RandomForestClassifier(n_estimators=200, max_depth=6,
                            random_state=random_state, n_jobs=-1)
rf.fit(x_train, y_cls_train)
rf_pred = rf.predict(x_test)
rf_proba = rf.predict_proba(x_test)[:, 1]
rf_auc = roc_auc_score(y_cls_test, rf_proba)
rf_acc = float((rf_pred == y_cls_test).mean())
rf_auc_mean, rf_auc_lo, rf_auc_hi = bootstrap_auc(y_cls_test.values, rf_proba)
rf_imp = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)

# model 2: gbm classifier
print("training gbm classifier...")
gbm_grid = {"max_depth": [3, 5], "learning_rate": [0.05, 0.1], "n_estimators": [100, 200]}
gbm_base = GradientBoostingClassifier(random_state=random_state)
gbm_cv = GridSearchCV(gbm_base, gbm_grid, cv=tscv5, scoring="roc_auc", n_jobs=-1)
gbm_cv.fit(x_train, y_cls_train)
gbm = gbm_cv.best_estimator_
gbm_pred = gbm.predict(x_test)
gbm_proba = gbm.predict_proba(x_test)[:, 1]
gbm_auc = roc_auc_score(y_cls_test, gbm_proba)
gbm_acc = float((gbm_pred == y_cls_test).mean())
gbm_auc_mean, gbm_auc_lo, gbm_auc_hi = bootstrap_auc(y_cls_test.values, gbm_proba)
gbm_imp = pd.Series(gbm.feature_importances_, index=feature_cols).sort_values(ascending=False)

# model 3: k-means regime detection
print("fitting k-means regimes...")
regime_data = df[["Returns", "Volume", "RSI"]].dropna()
scaler = StandardScaler()
x_scaled = scaler.fit_transform(regime_data)

sil_scores = {}
for k in range(2, 8):
    km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    lbl = km.fit_predict(x_scaled)
    sil_scores[k] = silhouette_score(x_scaled, lbl)

optimal_k = max(sil_scores, key=sil_scores.get)
best_sil = sil_scores[optimal_k]
km_final = KMeans(n_clusters=optimal_k, random_state=random_state, n_init=10)
regime_labels_all = km_final.fit_predict(x_scaled)
regime_data = regime_data.copy()
regime_data["Regime"] = regime_labels_all

# align regimes to test set
test_regimes = regime_data.loc[regime_data.index.isin(test.index), "Regime"]
test_regimes = test_regimes.reindex(test.index)

# regime stats
regime_stats = regime_data.groupby("Regime")["Returns"].agg(["mean", "std", "count"])
regime_stats.columns = ["AvgReturn", "Volatility", "Count"]
vol_stats = regime_data.groupby("Regime")["Volume"].mean().rename("AvgVolume")
regime_stats = regime_stats.join(vol_stats)

# transition matrix
reg_seq = regime_data["Regime"].values
trans_mat = pd.DataFrame(0, index=range(optimal_k), columns=range(optimal_k))
for i in range(len(reg_seq) - 1):
    trans_mat.loc[reg_seq[i], reg_seq[i+1]] += 1
trans_mat = trans_mat.div(trans_mat.sum(axis=1), axis=0).round(3)

# model 4: gbm regressor
print("training gbm regressor...")
reg_grid = {"n_estimators": [100, 200], "max_depth": [3, 5],
            "learning_rate": [0.05, 0.1], "subsample": [0.8, 1.0]}
gbm_reg_base = GradientBoostingRegressor(random_state=random_state)
gbm_reg_cv = GridSearchCV(gbm_reg_base, reg_grid, cv=tscv3,
                          scoring="neg_mean_squared_error", n_jobs=-1)
gbm_reg_cv.fit(x_train, y_reg_train)
gbm_reg = gbm_reg_cv.best_estimator_
reg_pred = gbm_reg.predict(x_test)
actual_ret = y_reg_test.values
reg_imp = pd.Series(gbm_reg.feature_importances_, index=feature_cols).sort_values(ascending=False)

# regression metrics
reg_mae = mean_absolute_error(actual_ret, reg_pred)
reg_rmse = float(np.sqrt(mean_squared_error(actual_ret, reg_pred)))
reg_mape = mape(actual_ret, reg_pred)
reg_r2 = r2_score(actual_ret, reg_pred)
reg_dir = float(np.mean(np.sign(actual_ret) == np.sign(reg_pred)) * 100)

naive_zero = np.zeros_like(actual_ret)
naive_persist = np.concatenate([[0], actual_ret[:-1]])
naive_mae = mean_absolute_error(actual_ret, naive_zero)
naive_rmse = float(np.sqrt(mean_squared_error(actual_ret, naive_zero)))

reg_beats_naive = reg_mae < naive_mae

# dm test
dm_stat, dm_pval = diebold_mariano(actual_ret, reg_pred, naive_zero)

# naive classification baseline
majority_class = int(y_cls_train.mode()[0])
naive_cls_acc = float((y_cls_test == majority_class).mean())

# mcnemar test
mc_chi2, mc_pval = mcnemar_test(y_cls_test.values, rf_pred, gbm_pred)

# agreement analysis
agreement_mask = rf_pred == gbm_pred
agreement_pct = float(agreement_mask.mean() * 100)
disagreement_pct = 100 - agreement_pct

print(f"all models trained. agreement rf vs gbm: {agreement_pct:.1f}%")

# step 2 - generate all plots
section("step 2 - generating plots")

palette = {"RF": "#2196F3", "GBM": "#FF5722", "Reg": "#4CAF50",
           "Naive": "#9E9E9E", "Regime0": "#E91E63", "Regime1": "#00BCD4",
           "Regime2": "#FF9800", "Regime3": "#9C27B0"}
reg_colors = [palette.get(f"Regime{i}", f"C{i}") for i in range(optimal_k)]

# plot 1: side-by-side confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, pred, title, cmap in zip(
    axes,
    [rf_pred, gbm_pred],
    ["random forest classifier", "gbm classifier"],
    ["Blues", "Oranges"],
):
    cm = confusion_matrix(y_cls_test, pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, ax=ax,
                xticklabels=["down (0)", "up (1)"],
                yticklabels=["down (0)", "up (1)"],
                linewidths=0.5, cbar=False)
    ax.set_title(title, fontweight="bold", fontsize=12)
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
fig.suptitle("confusion matrices - classification models", fontsize=14, fontweight="bold")
plt.tight_layout()
img_cm = save_fig(fig, "01_confusion_matrices.png")
print(f"  saved: {img_cm}")

# plot 2: roc curves overlaid
fig, ax = plt.subplots(figsize=(7, 6))
for proba, label, color, auc_val, lo, hi in [
    (rf_proba,  f"random forest (auc={rf_auc:.3f})",  palette["RF"],  rf_auc,  rf_auc_lo,  rf_auc_hi),
    (gbm_proba, f"gbm classifier (auc={gbm_auc:.3f})", palette["GBM"], gbm_auc, gbm_auc_lo, gbm_auc_hi),
]:
    fpr, tpr, _ = roc_curve(y_cls_test, proba)
    ax.plot(fpr, tpr, linewidth=2, color=color, label=f"{label}\n95% ci [{lo:.3f} to {hi:.3f}]")
ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="random baseline")
ax.fill_between([0, 1], [0, 1], alpha=0.05, color="grey")
ax.set_xlabel("false positive rate", fontsize=12)
ax.set_ylabel("true positive rate", fontsize=12)
ax.set_title("roc curves - classification models", fontweight="bold", fontsize=13)
ax.legend(fontsize=10, loc="lower right")
ax.grid(alpha=0.25)
plt.tight_layout()
img_roc = save_fig(fig, "02_roc_curves.png")
print(f"  saved: {img_roc}")

# plot 3: precision/recall/f1 comparison bar chart
metrics_df = pd.DataFrame(index=["precision", "recall", "f1-score"])
for label, pred in [("random forest", rf_pred), ("gbm classifier", gbm_pred)]:
    p, r, f, _ = precision_recall_fscore_support(y_cls_test, pred, average="macro")
    metrics_df[label] = [p, r, f]

fig, ax = plt.subplots(figsize=(8, 4))
x = np.arange(len(metrics_df))
w = 0.32
bars1 = ax.bar(x - w/2, metrics_df["random forest"], w,
               label="random forest", color=palette["RF"], alpha=0.85)
bars2 = ax.bar(x + w/2, metrics_df["gbm classifier"], w,
               label="gbm classifier", color=palette["GBM"], alpha=0.85)
for bars in [bars1, bars2]:
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(metrics_df.index, fontsize=12)
ax.set_ylim(0, 1.08)
ax.set_ylabel("score")
ax.set_title("precision / recall / f1 - macro averaged", fontweight="bold")
ax.legend()
ax.grid(axis="y", alpha=0.25)
plt.tight_layout()
img_prf = save_fig(fig, "03_precision_recall_f1.png")
print(f"  saved: {img_prf}")

# plot 4: regression - actual vs predicted scatter
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(actual_ret, reg_pred, alpha=0.4, s=14, color=palette["Reg"])
lo, hi = min(actual_ret.min(), reg_pred.min()), max(actual_ret.max(), reg_pred.max())
ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.5, label="perfect prediction")
ax.set_xlabel("actual return (%)", fontsize=12)
ax.set_ylabel("predicted return (%)", fontsize=12)
ax.set_title("regression - actual vs predicted returns", fontweight="bold")
ax.legend()
ax.grid(alpha=0.25)
plt.tight_layout()
img_reg_scatter = save_fig(fig, "04_reg_scatter.png")
print(f"  saved: {img_reg_scatter}")

# plot 5: residual distribution
residuals = actual_ret - reg_pred
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.hist(residuals, bins=45, color=palette["Reg"], edgecolor="white", alpha=0.85)
ax1.axvline(0, color="red", linestyle="--", linewidth=1.5)
ax1.set_title("residuals distribution", fontweight="bold")
ax1.set_xlabel("residual (%)")
ax1.set_ylabel("frequency")
(osm, osr), (slope, intercept, _) = stats.probplot(residuals, dist="norm")
ax2.scatter(osm, osr, s=8, alpha=0.5, color=palette["Reg"])
ax2.plot(osm, slope * np.array(osm) + intercept, "r-", linewidth=1.5)
ax2.set_title("q-q plot of residuals", fontweight="bold")
ax2.set_xlabel("theoretical quantiles")
ax2.set_ylabel("sample quantiles")
ax2.grid(alpha=0.25)
fig.suptitle("residual diagnostics - regression model", fontsize=13, fontweight="bold")
plt.tight_layout()
img_resid = save_fig(fig, "05_residuals.png")
print(f"  saved: {img_resid}")

# plot 6: regime time series
fig, ax = plt.subplots(figsize=(14, 5))
price_full = df["Price"]
ax.plot(price_full.index, price_full.values, color="lightgrey", linewidth=0.9, zorder=1)
for r in range(optimal_k):
    mask = regime_data["Regime"] == r
    avg = regime_stats.loc[r, "AvgReturn"]
    ax.scatter(regime_data.index[mask], df.loc[regime_data.index[mask], "Price"],
               color=reg_colors[r], s=7, alpha=0.6, zorder=2,
               label=f"regime {r} (avg={avg:+.2f}%)")
ax.set_title("ogdc price - market regimes (k-means)", fontweight="bold", fontsize=13)
ax.set_xlabel("date")
ax.set_ylabel("price (pkr)")
ax.legend(fontsize=9)
ax.grid(alpha=0.2)
plt.tight_layout()
img_regime_ts = save_fig(fig, "06_regime_timeseries.png")
print(f"  saved: {img_regime_ts}")

# plot 7: model agreement heatmap
agree_df = pd.DataFrame({
    "RF": rf_pred,
    "GBM": gbm_pred,
    "Actual": y_cls_test.values,
})
agree_matrix = pd.crosstab(agree_df["RF"], agree_df["GBM"],
                           rownames=["random forest"], colnames=["gbm classifier"])
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(agree_matrix, annot=True, fmt="d", cmap="YlOrRd", ax=ax, linewidths=0.5)
ax.set_title("rf vs gbm prediction agreement\n(rows=rf, cols=gbm)", fontweight="bold")
plt.tight_layout()
img_agree = save_fig(fig, "07_model_agreement.png")
print(f"  saved: {img_agree}")

# plot 8: feature importance comparison
top_n = 10
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
for ax, imp, title, color in zip(
    axes,
    [rf_imp, gbm_imp, reg_imp],
    ["random forest\n(classifier)", "gbm\n(classifier)", "gbm\n(regressor)"],
    [palette["RF"], palette["GBM"], palette["Reg"]],
):
    imp.head(top_n).plot(kind="bar", ax=ax, color=color, edgecolor="white", alpha=0.9)
    ax.set_title(title, fontweight="bold", fontsize=11)
    ax.set_ylabel("importance")
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.grid(axis="y", alpha=0.25)
fig.suptitle("feature importance - top 10 per model", fontsize=13, fontweight="bold")
plt.tight_layout()
img_feat = save_fig(fig, "08_feature_importance.png")
print(f"  saved: {img_feat}")

# plot 9: classification errors by regime
test_aligned = test.copy()
test_aligned["Regime"] = test_regimes
test_aligned["rf_pred"] = rf_pred
test_aligned["gbm_pred"] = gbm_pred
test_aligned["rf_err"] = (rf_pred != y_cls_test.values).astype(int)
test_aligned["gbm_err"] = (gbm_pred != y_cls_test.values).astype(int)
test_aligned["reg_err"] = np.abs(actual_ret - reg_pred)

regime_errors = test_aligned.groupby("Regime")[["rf_err", "gbm_err"]].mean() * 100
fig, ax = plt.subplots(figsize=(6, 4))
regime_errors.plot(kind="bar", ax=ax, color=[palette["RF"], palette["GBM"]],
                   edgecolor="white", alpha=0.85)
ax.set_title("classification error rate by regime", fontweight="bold")
ax.set_xlabel("regime")
ax.set_ylabel("error rate (%)")
ax.set_xticklabels([f"regime {int(r)}" for r in regime_errors.index], rotation=0)
ax.legend(["random forest", "gbm classifier"])
ax.grid(axis="y", alpha=0.25)
plt.tight_layout()
img_regime_err = save_fig(fig, "09_errors_by_regime.png")
print(f"  saved: {img_regime_err}")

# plot 10: regression time series (test period)
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(test.index, actual_ret, label="actual returns", linewidth=1.2, color="steelblue")
ax.plot(test.index, reg_pred, label="predicted", linewidth=1.0, color="tomato",
        linestyle="--", alpha=0.8)
ax.fill_between(test.index, actual_ret, reg_pred, alpha=0.12, color="grey")
ax.axhline(0, color="black", linewidth=0.7, alpha=0.4)
ax.set_title("regression - actual vs predicted returns (test period)", fontweight="bold")
ax.set_xlabel("date")
ax.set_ylabel("return (%)")
ax.legend()
ax.grid(alpha=0.2)
plt.tight_layout()
img_reg_ts = save_fig(fig, "10_reg_timeseries.png")
print(f"  saved: {img_reg_ts}")

print("\nall plots saved.")

# step 3 - failure analysis
section("step 3 - failure analysis")

both_cls_wrong = (rf_pred != y_cls_test.values) & (gbm_pred != y_cls_test.values)
reg_large_err = np.abs(actual_ret - reg_pred) > (2 * reg_mae)
all_fail_mask = both_cls_wrong & reg_large_err
fail_df = test_aligned[all_fail_mask].copy()
fail_df["actual_return"] = actual_ret[all_fail_mask]
fail_df["reg_error"] = (actual_ret - reg_pred)[all_fail_mask]
fail_df["abs_return"] = np.abs(fail_df["actual_return"])

print(f"dates where all models failed: {len(fail_df)}")
if len(fail_df) > 0:
    print(fail_df[["Price", "actual_return", "reg_error", "Regime"]].to_string())

# step 4 - build markdown report
section("step 4 - building markdown report")

# feature overlap table
top10_rf = set(rf_imp.head(10).index)
top10_gbm = set(gbm_imp.head(10).index)
top10_reg = set(reg_imp.head(10).index)
all_top = sorted(top10_rf | top10_gbm | top10_reg)

overlap_rows = []
for feat in all_top:
    overlap_rows.append({
        "feature": feat,
        "rf": "x" if feat in top10_rf else "",
        "gbm_cls": "x" if feat in top10_gbm else "",
        "gbm_reg": "x" if feat in top10_reg else "",
        "count": sum([feat in top10_rf, feat in top10_gbm, feat in top10_reg]),
    })
overlap_df = pd.DataFrame(overlap_rows).sort_values("count", ascending=False)

# overall checks
best_clf = "random forest" if rf_auc >= gbm_auc else "gbm classifier"
best_clf_auc = max(rf_auc, gbm_auc)
best_clf_acc = max(rf_acc, gbm_acc)
all_below_base = best_clf_acc < naive_cls_acc and not reg_beats_naive
unstable_regimes = best_sil < 0.1

rf_beats = rf_acc > naive_cls_acc
gbm_beats = gbm_acc > naive_cls_acc

# write report
rpt(
    f"# ogdc stock prediction - model comparison report",
    f"",
    f"> generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    f"> test period: {test_start} to {test_end} ({n_test} trading days)",
    f"> note: xgboost was unavailable. scikit-learn's gradientboosting was used as drop-in replacement.",
    f"",
)

# 1. executive summary
rpt("---", "## 1. executive summary", "")

if all_below_base:
    rpt("> warning: no predictive power found - all models performed at or below baseline.", "")

rpt(
    "| model | type | key metric | value | beats baseline? |",
    "|-------|------|-----------|-------|----------------|",
    f"| random forest | classification | roc-auc | {rf_auc:.4f} | {'yes' if rf_beats else 'no'} |",
    f"| gbm classifier | classification | roc-auc | {gbm_auc:.4f} | {'yes' if gbm_beats else 'no'} |",
    f"| gbm regressor | regression | r2 | {reg_r2:.4f} | {'yes' if reg_beats_naive else 'no'} |",
    f"| k-means | unsupervised | silhouette score | {best_sil:.4f} | n/a |",
    "",
    f"naive classification baseline (always predict majority class {majority_class}): "
    f"accuracy = {naive_cls_acc:.4f}",
    f"naive regression baseline (always predict 0% return): mae = {naive_mae:.4f}%",
    "",
)

if not rf_beats:
    rpt("> random forest does not beat the naive majority-class baseline.", "")
if not gbm_beats:
    rpt("> gbm classifier does not beat the naive majority-class baseline.", "")
if not reg_beats_naive:
    rpt("> gbm regressor does not beat the naive zero-return baseline.", "")
if best_sil < 0.1:
    rpt("> silhouette score < 0.1 - regimes are unstable. consider reducing k.", "")

# 2. classification comparison
rpt(
    "---",
    "## 2. classification models comparison",
    "",
    "### confusion matrices",
    "",
    f"![confusion matrices]({img_cm})",
    "",
    "### roc curves",
    "",
    f"![roc curves]({img_roc})",
    "",
    "### precision / recall / f1",
    "",
    f"![prf1 chart]({img_prf})",
    "",
    "| metric | random forest | gbm classifier | delta (rf - gbm) |",
    "|--------|--------------|----------------|------------------|",
)
for metric in ["precision", "recall", "f1-score"]:
    rf_val = float(metrics_df.loc[metric, "random forest"])
    gbm_val = float(metrics_df.loc[metric, "gbm classifier"])
    delta = rf_val - gbm_val
    rpt(f"| {metric} | {rf_val:.4f} | {gbm_val:.4f} | {delta:+.4f} |")

rpt(
    f"| accuracy | {rf_acc:.4f} | {gbm_acc:.4f} | {rf_acc-gbm_acc:+.4f} |",
    f"| roc-auc | {rf_auc:.4f} | {gbm_auc:.4f} | {rf_auc-gbm_auc:+.4f} |",
    "",
    f"winner: {best_clf} (roc-auc = {best_clf_auc:.4f})",
    "",
    f"the margin is {abs(rf_auc - gbm_auc):.4f} auc points - ",
)

# 3. regression model
rpt(
    "---",
    "## 3. regression model performance",
    "",
    "### actual vs predicted returns",
    "",
    f"![scatter]({img_reg_scatter})",
    "",
    f"![time series]({img_reg_ts})",
    "",
    "### residual diagnostics",
    "",
    f"![residuals]({img_resid})",
    "",
    "### metrics summary",
    "",
    "| metric | model | naive baseline (zero) | better? |",
    "|--------|-------|-----------------------|---------|",
    f"| mae | {reg_mae:.4f}% | {naive_mae:.4f}% | {'yes' if reg_mae < naive_mae else 'no'} |",
    f"| rmse | {reg_rmse:.4f}% | {naive_rmse:.4f}% | {'yes' if reg_rmse < naive_rmse else 'no'} |",
    f"| mape | {reg_mape:.2f}% | - | - |",
    f"| r2 | {reg_r2:.4f} | 0.0000 | {'yes' if reg_r2 > 0 else 'no'} |",
    f"| directional accuracy | {reg_dir:.2f}% | 50.00% | {'yes' if reg_dir > 50 else 'no'} |",
    "",
)
if reg_r2 < 0:
    rpt("> r2 is negative - the model is worse than predicting the mean return.", "")
elif reg_r2 < 0.05:
    rpt(f"> r2 = {reg_r2:.4f} - low but realistic for daily financial returns.", "")
else:
    rpt(f"> r2 = {reg_r2:.4f} - solid explanatory power.", "")

if reg_mape > 100:
    rpt("> mape > 100% - returns are highly volatile; use mae/rmse as primary metrics.", "")

# 4. regime analysis
best_regime = int(regime_stats["AvgReturn"].idxmax())
rpt(
    "---",
    "## 4. regime analysis (k-means)",
    "",
    f"optimal k: {optimal_k} regimes | best silhouette score: {best_sil:.4f}",
    "",
)
if unstable_regimes:
    rpt("> silhouette score < 0.1 - regimes are unstable. consider reducing k.", "")

rpt(
    "### regime characteristics",
    "",
    "| regime | avg return (%) | volatility (sigma) | avg volume | day count | best? |",
    "|--------|---------------|-------------------|------------|-----------|-------|",
)
for r in range(optimal_k):
    s = regime_stats.loc[r]
    best_flag = "*" if r == best_regime else ""
    rpt(f"| regime {r} | {s['AvgReturn']:+.4f} | {s['Volatility']:.4f} | "
        f"{s['AvgVolume']:,.0f} | {int(s['Count'])} | {best_flag} |")

rpt(
    "",
    f"best regime: regime {best_regime} (highest average return = "
    f"{regime_stats.loc[best_regime, 'AvgReturn']:+.4f}%)",
    "",
    "### regime transition matrix",
    "",
    "probability of transitioning from one regime to another (rows = current, cols = next):",
    "",
    "| from to | " + " | ".join([f"regime {c}" for c in trans_mat.columns]) + " |",
    "|" + "---|" * (optimal_k + 1),
)
for r in trans_mat.index:
    row_vals = " | ".join([f"{trans_mat.loc[r, c]:.3f}" for c in trans_mat.columns])
    rpt(f"| regime {r} | {row_vals} |")

rpt(
    "",
    "### price coloured by regime",
    "",
    f"![regime time series]({img_regime_ts})",
    "",
)

# 5. model agreement
rpt(
    "---",
    "## 5. model agreement analysis",
    "",
    f"rf vs gbm classifier agreement: {agreement_pct:.1f}%",
    "",
)
if disagreement_pct > 30:
    rpt(f"> models disagree on {disagreement_pct:.1f}% of predictions - above the 30% threshold.")
    rpt("> an ensemble is likely to reduce errors.", "")
else:
    rpt(f"> models agree on {agreement_pct:.1f}% of predictions.", "")

# high-confidence agreement
high_conf_rf = np.abs(rf_proba - 0.5) > 0.3
high_conf_gbm = np.abs(gbm_proba - 0.5) > 0.3
both_high = high_conf_rf & high_conf_gbm
if both_high.sum() > 0:
    avg_reg_err_hc = float(np.abs(actual_ret[both_high] - reg_pred[both_high]).mean())
    avg_reg_err_lc = float(np.abs(actual_ret[~both_high] - reg_pred[~both_high]).mean()) if (~both_high).sum() > 0 else float("nan")
    rpt(
        "### high-confidence classification vs regression error",
        "",
        "| classifier confidence | n samples | avg abs regression error |",
        "|----------------------|-----------|--------------------------|",
        f"| both models high-conf (|p-0.5|>0.3) | {int(both_high.sum())} | {avg_reg_err_hc:.4f}% |",
        f"| otherwise | {int((~both_high).sum())} | {avg_reg_err_lc:.4f}% |",
        "",
    )

rpt(
    f"![agreement matrix]({img_agree})",
    "",
    "### classification errors by regime",
    "",
    f"![errors by regime]({img_regime_err})",
    "",
    "| regime | rf error rate (%) | gbm error rate (%) |",
    "|--------|------------------|--------------------|",
)
for r in regime_errors.index:
    rpt(f"| regime {int(r)} | {regime_errors.loc[r, 'rf_err']:.2f} | {regime_errors.loc[r, 'gbm_err']:.2f} |")
rpt("")

# 6. statistical significance
rpt(
    "---",
    "## 6. statistical significance tests",
    "",
    "### mcnemar's test - are the two classifiers significantly different?",
    "",
    f"| statistic | value |",
    f"|-----------|-------|",
    f"| chi2 | {mc_chi2:.4f} |",
    f"| p-value | {mc_pval:.4f} |",
    f"| significant (p<0.05)? | {'yes' if mc_pval < 0.05 else 'no'} |",
    "",
)
if mc_pval < 0.05:
    rpt("> the two classifiers are statistically significantly different (p < 0.05).")
    rpt(f"> {best_clf} is the superior classifier.", "")
else:
    rpt("> the two classifiers are not significantly different (p >= 0.05).")
    rpt("> either model can be deployed; prefer the simpler/faster one.", "")

rpt(
    "### diebold-mariano test - regression vs naive zero baseline",
    "",
    f"| statistic | value |",
    f"|-----------|-------|",
    f"| dm statistic | {dm_stat:.4f} |",
    f"| p-value | {dm_pval:.4f} |",
    f"| significant (p<0.05)? | {'yes' if dm_pval < 0.05 else 'no'} |",
    "",
)
if dm_pval < 0.05:
    winner_dm = "gbm regressor" if dm_stat < 0 else "naive baseline"
    rpt(f"> statistically significant difference. {winner_dm} is more accurate.", "")
else:
    rpt("> no statistically significant difference - the regression model provides no provable", "")
    rpt("> improvement over predicting 0% return every day.", "")

rpt(
    "### bootstrap 95% confidence intervals for roc-auc",
    "",
    f"| model | mean auc | 95% ci lower | 95% ci upper |",
    f"|-------|----------|-------------|-------------|",
    f"| random forest | {rf_auc_mean:.4f} | {rf_auc_lo:.4f} | {rf_auc_hi:.4f} |",
    f"| gbm classifier | {gbm_auc_mean:.4f} | {gbm_auc_lo:.4f} | {gbm_auc_hi:.4f} |",
    "",
)

# 7. feature importance
rpt(
    "---",
    "## 7. feature importance comparison",
    "",
    f"![feature importances]({img_feat})",
    "",
    "### top-10 feature overlap table",
    "",
    "| feature | rf classifier | gbm classifier | gbm regressor | models sharing |",
    "|---------|:------------:|:--------------:|:-------------:|:--------------:|",
)
for _, row in overlap_df.iterrows():
    rpt(f"| {row['feature']} | {row['rf']} | {row['gbm_cls']} | {row['gbm_reg']} | {row['count']}/3 |")

all3 = overlap_df[overlap_df["count"] == 3]["feature"].tolist()
rpt(
    "",
    f"features important to all 3 models: {', '.join(all3) if all3 else 'none'}",
    "",
)

# 8. failure analysis
rpt(
    "---",
    "## 8. failure analysis",
    "",
    f"failure criterion: both classifiers wrong and regression error > 2x mae ({2*reg_mae:.4f}%)",
    "",
    f"total failure dates: {len(fail_df)} out of {n_test} test days "
    f"({100*len(fail_df)/n_test:.1f}%)",
    "",
)
if len(fail_df) > 0:
    rpt("### dates where all models failed", "")
    rpt("| date | price | actual return (%) | reg error (%) | regime | volatility context |")
    rpt("|------|-------|------------------|--------------|--------|--------------------|")
    for dt, row in fail_df.head(20).iterrows():
        vol_flag = "high vol" if abs(row.get("actual_return", 0)) > 3 else "normal"
        rpt(
            f"| {dt.strftime('%Y-%m-%d')} | "
            f"{row.get('Price', 'n/a'):.2f} | "
            f"{row.get('actual_return', 0):+.2f} | "
            f"{abs(row.get('reg_error', 0)):.4f} | "
            f"regime {int(row.get('Regime', -1))} | "
            f"{vol_flag} |"
        )
    if len(fail_df) > 20:
        rpt(f"\n*(showing first 20 of {len(fail_df)} failure dates)*")

    avg_fail_vol = float(fail_df["actual_return"].abs().mean())
    avg_test_vol = float(np.abs(actual_ret).mean())
    rpt(
        "",
        f"average |return| on failure dates: {avg_fail_vol:.4f}%",
        f"average |return| on all test dates: {avg_test_vol:.4f}%",
        "",
    )
    if avg_fail_vol > 1.5 * avg_test_vol:
        rpt("> failures cluster on high-volatility days - likely driven by unexpected events", "")
        rpt("> (central bank decisions, earnings surprises, geopolitical shocks).", "")
        rpt("> consider adding a volatility regime filter before generating signals.", "")
else:
    rpt("> no dates found where all models failed simultaneously.", "")

# 9. conclusion and recommendation
rpt(
    "---",
    "## 9. conclusion and recommendation",
    "",
    "### model selection",
    "",
)

if all_below_base:
    deploy_rec = "no model recommended for deployment. all models failed to beat baselines."
    ensemble_rec = "ensembling would not help - the signal itself is absent."
elif rf_auc >= gbm_auc:
    deploy_rec = (
        f"deploy random forest classifier for direction prediction.\n"
        f"it achieved roc-auc = {rf_auc:.4f} vs naive accuracy = {naive_cls_acc:.4f}, "
        f"with bootstrap 95% ci [{rf_auc_lo:.3f} to {rf_auc_hi:.3f}]."
    )
else:
    deploy_rec = (
        f"deploy gbm classifier for direction prediction.\n"
        f"it achieved roc-auc = {gbm_auc:.4f} vs naive accuracy = {naive_cls_acc:.4f}, "
        f"with bootstrap 95% ci [{gbm_auc_lo:.3f} to {gbm_auc_hi:.3f}]."
    )

if disagreement_pct > 30:
    ensemble_rec = (
        f"ensemble is recommended. rf and gbm disagree on {disagreement_pct:.1f}% of predictions. "
        f"a probability-averaged ensemble is likely to improve robustness."
    )
elif mc_pval > 0.05:
    ensemble_rec = (
        f"ensemble is optional. the classifiers are not significantly different (mcnemar p={mc_pval:.3f}), "
        f"so ensembling may offer marginal gain at the cost of complexity."
    )
else:
    ensemble_rec = (
        f"ensemble may help. the classifiers are statistically different - "
        f"combining their probability outputs could reduce variance."
    )

if reg_r2 > 0.01 and reg_beats_naive:
    reg_rec = (
        f"the regression model (r2={reg_r2:.4f}, da={reg_dir:.1f}%) adds value beyond "
        f"pure direction calls - use it for position sizing (scale exposure proportional to "
        f"predicted return magnitude)."
    )
else:
    reg_rec = (
        f"the regression model (r2={reg_r2:.4f}) provides limited incremental value over "
        f"the classification approach. stick to direction signals for live trading."
    )

if best_clf_acc > naive_cls_acc + 0.05:
    conclusion = (
        f"project conclusion: statistically meaningful price-direction predictability "
        f"was found in ogdc stock returns, with autocorrelation in returns enabling a "
        f"{best_clf} to exceed the naive baseline by "
        f"{best_clf_acc - naive_cls_acc:.1%} in accuracy."
    )
elif best_clf_acc > naive_cls_acc:
    conclusion = (
        f"project conclusion: weak but measurable price-direction predictability exists in ogdc "
        f"returns; the {best_clf} marginally outperforms random guessing, consistent with "
        f"a weak-form semi-efficient market."
    )
else:
    conclusion = (
        "project conclusion: no robust predictive signal was found - ogdc daily returns "
        "appear to follow a near-random walk, consistent with the efficient market hypothesis."
    )

rpt(
    deploy_rec, "",
    "### ensemble recommendation", "",
    ensemble_rec, "",
    "### regression model usage", "",
    reg_rec, "",
    "### project conclusion", "",
    conclusion, "",
    "---",
    "",
    f"report generated by ogdc_model_comparison.py on {datetime.now().strftime('%Y-%m-%d')}.",
    "",
)

# step 5 - save report
section("step 5 - saving report")
report_text = "\n".join(_report_lines)
with open(report_path, "w", encoding="utf-8") as f:
    f.write(report_text)
print(f"\nmarkdown report saved to {report_path}")
print(f"plot embeds in report: {len([l for l in _report_lines if l.startswith('!')])}")
print(f"report length: {len(report_text):,} characters")