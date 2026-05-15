"""
ogdc_enhanced_ml.py
===================
enhanced ml pipeline — combines trend features (bb, ma, volatility)
with the original feature set from ogdc_analysis.py

compares:
  round 1 — original features only  (lags, rolling stats, rsi, macd, date)
  round 2 — enhanced features       (+ bollinger bands, moving averages, volatility)

models:
  • random forest classifier     — direction (up/down)
  • gbm classifier               — direction (up/down)
  • gbm regressor                — exact return %

for each round:
  • timeseriessplit cross-validation
  • test-set classification report + roc-auc
  • test-set regression mae / rmse / r² / directional accuracy
  • feature importance (top 20)
  • head-to-head comparison table

outputs: 5 png charts + ogdc_enhanced_results.csv
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import (
    roc_auc_score, classification_report, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score
)

# hardcoded paths for input data, outputs, and images
base = os.path.dirname(os.path.abspath(__file__))  # scripts folder
data_dir = os.path.join(base, "..", "data", "processed")
os.makedirs(data_dir, exist_ok=True)

regime_path = os.path.join(data_dir, "ogdc_with_regimes.csv")  # from ogdc_analysis
trend_path = os.path.join(data_dir, "ogdc_trend_features.csv")  # from trend analysis
out_csv = os.path.join(data_dir, "ogdc_enhanced_results.csv")

img_dir = os.path.join(base, "..", "outputs", "images")
os.makedirs(img_dir, exist_ok=True)
img_pre = os.path.join(img_dir, "eml_")

# dark theme 
plt.rcParams.update({
    "figure.facecolor":"#0d1117","axes.facecolor":"#161b22",
    "axes.edgecolor":"#30363d","axes.labelcolor":"#e6edf3",
    "xtick.color":"#8b949e","ytick.color":"#8b949e",
    "text.color":"#e6edf3","grid.color":"#21262d","grid.linewidth":0.6,
    "legend.facecolor":"#161b22","legend.edgecolor":"#30363d",
})
c = {"orig":"#58a6ff","enh":"#3fb950","rf":"#f0883e","gbm":"#d2a8ff",
     "reg":"#ffa657","imp":"#388bfd","delta_pos":"#3fb950","delta_neg":"#f85149"}

random_state = 42
tscv5 = TimeSeriesSplit(n_splits=5)  # 5-fold for classification
tscv3 = TimeSeriesSplit(n_splits=3)  # 3-fold for regression (faster)

# save figure helper (images go to outputs/images/)
def save_fig(fig, name):
    path = img_pre + name
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  → saved: eml_{name}")

# console formatting helpers
def banner(t): print(f"\n{'═'*70}\n  {t}\n{'═'*70}")
def sub(t):    print(f"\n  {'─'*58}\n  {t}\n  {'─'*58}")

# mean absolute percentage error (avoids division by zero)
def mape(actual, pred):
    mask = actual != 0
    return float(np.mean(np.abs((actual[mask]-pred[mask])/actual[mask]))*100)

# ══════════════════════════════════════════════════════════════════════════════
# load & merge data from both analysis outputs
# ══════════════════════════════════════════════════════════════════════════════
banner("loading & merging data")

regime_df = pd.read_csv(regime_path, index_col="Date", parse_dates=True)
trend_df  = pd.read_csv(trend_path,  index_col="Date", parse_dates=True)

# merge on date index (inner join keeps only common dates)
merged = regime_df.join(trend_df, rsuffix="_t", how="inner")
merged.dropna(subset=["Returns","Direction"], inplace=True)
merged.sort_index(inplace=True)

print(f"  regime df : {regime_df.shape}")
print(f"  trend df  : {trend_df.shape}")
print(f"  merged    : {merged.shape}  ({merged.index.min().date()} → {merged.index.max().date()})")

# original features from ogdc_analysis (technical indicators)
orig_features = [
    "Return_lag1","Return_lag2","Return_lag3","Return_lag5",
    "Return_roll_mean5","Return_roll_std5","Return_roll_mean10","Return_roll_std10",
    "RSI","MACD","MACD_signal","MACD_hist",
    "DayOfWeek","Month","Quarter",
    "Open","High","Low","Volume","ChangeP",
]

# new features from trend analysis (bollinger bands, moving averages, volatility)
new_features = [
    # moving averages
    "SMA_20","SMA_50","SMA_200","EMA_12","EMA_26","WMA_20",
    # bollinger bands components
    "BB_mid","BB_upper","BB_lower",
    # bollinger band derived indicators
    "BB_pct_b","BB_bandwidth","BB_squeeze",
    "BB_sig_breakout_long","BB_sig_breakout_short","BB_sig_mr_buy","BB_sig_mr_sell",
    # volatility estimators
    "HV_20","HV_60","HV_Parkinson","HV_GarmanKlass",
]

# keep only columns that exist and have sufficient non-null values
orig_features = [c for c in orig_features if c in merged.columns and merged[c].notna().sum() > 100]
new_features  = [c for c in new_features  if c in merged.columns and merged[c].notna().sum() > 100]
enhanced_features = orig_features + new_features

# add price-vs-ma ratio features (normalised percentage difference)
for ma in ["SMA_20","SMA_50","SMA_200","EMA_12","EMA_26"]:
    if ma in merged.columns:
        col = f"Price_vs_{ma}"
        merged[col] = (merged["Price"] / merged[ma] - 1) * 100  # % deviation
        enhanced_features.append(col)

# drop rows with any missing values in features or targets
merged_clean = merged.dropna(subset=enhanced_features + ["Direction","Returns"])

print(f"  original features   : {len(orig_features)}")
print(f"  new trend features  : {len(new_features)}")
print(f"  total enhanced      : {len(enhanced_features)}")
print(f"  rows after dropna   : {len(merged_clean)}")

# chronological train/test split (80/20, no shuffling)
split_idx   = int(len(merged_clean) * 0.80)
train       = merged_clean.iloc[:split_idx]
test        = merged_clean.iloc[split_idx:]

print(f"  train: {len(train)}  test: {len(test)}")

# separate features and targets for both feature sets
x_tr_orig = train[orig_features];       x_te_orig = test[orig_features]
x_tr_enh  = train[enhanced_features];   x_te_enh  = test[enhanced_features]
y_cls_tr  = train["Direction"].astype(int)  # binary classification target
y_cls_te  = test["Direction"].astype(int)
y_reg_tr  = train["Returns"]  # continuous regression target
y_reg_te  = test["Returns"]

# ══════════════════════════════════════════════════════════════════════════════
# training & evaluation helpers
# ══════════════════════════════════════════════════════════════════════════════

def eval_clf(model, x_te, y_te, label):
    """evaluate classifier: returns accuracy, roc-auc, predictions"""
    pred   = model.predict(x_te)
    proba  = model.predict_proba(x_te)[:,1]  # probability of positive class
    acc    = float((pred == y_te).mean())
    auc    = roc_auc_score(y_te, proba)  # area under roc curve (0.5=random, 1=perfect)
    return {"label":label,"acc":acc,"auc":auc,"pred":pred,"proba":proba}

def eval_reg(model, x_te, y_te, label):
    """evaluate regressor: returns mae, rmse, r², directional accuracy, mape"""
    pred = model.predict(x_te)
    act  = y_te.values
    return {
        "label": label,
        "mae":   mean_absolute_error(act, pred),  # mean absolute error in %
        "rmse":  float(np.sqrt(mean_squared_error(act, pred))),  # root mean squared error
        "r2":    r2_score(act, pred),  # variance explained (0 to 1, can be negative)
        "da":    float(np.mean(np.sign(act)==np.sign(pred))*100),  # directional accuracy %
        "mape":  mape(act, pred),
        "pred":  pred,
        "actual":act,
    }

clf_results = {}
reg_results = {}
feat_importances = {}

for label, x_tr, x_te in [
    ("original",  x_tr_orig, x_te_orig),
    ("enhanced",  x_tr_enh,  x_te_enh),
]:
    sub(f"round: {label} features")

    # ── random forest classifier (ensemble of 200 trees) ──────────────────────
    print(f"  training rf classifier ({label})...")
    rf = RandomForestClassifier(n_estimators=200, max_depth=6,
                                 random_state=random_state, n_jobs=-1)
    rf.fit(x_tr, y_cls_tr)
    res = eval_clf(rf, x_te, y_cls_te, f"rf [{label}]")
    clf_results[f"rf_{label}"] = res
    feat_importances[f"rf_{label}"] = pd.Series(rf.feature_importances_,
                                                  index=x_tr.columns).sort_values(ascending=False)
    print(f"    accuracy={res['acc']:.4f}  roc-auc={res['auc']:.4f}")

    # ── gbm classifier (gradient boosting with grid search) ──────────────────
    print(f"  tuning gbm classifier ({label})...")
    gbm_grid = {"max_depth":[3,5],"learning_rate":[0.05,0.1],"n_estimators":[100,200]}
    gbm = GridSearchCV(GradientBoostingClassifier(random_state=random_state),
                       gbm_grid, cv=tscv5, scoring="roc_auc", n_jobs=-1)
    gbm.fit(x_tr, y_cls_tr)
    res_g = eval_clf(gbm.best_estimator_, x_te, y_cls_te, f"gbm [{label}]")
    clf_results[f"gbm_{label}"] = res_g
    feat_importances[f"gbm_{label}"] = pd.Series(
        gbm.best_estimator_.feature_importances_, index=x_tr.columns).sort_values(ascending=False)
    print(f"    accuracy={res_g['acc']:.4f}  roc-auc={res_g['auc']:.4f}  best={gbm.best_params_}")

    # ── gbm regressor (predicts exact return percentage) ─────────────────────
    print(f"  tuning gbm regressor ({label})...")
    reg_grid = {"n_estimators":[100,200],"max_depth":[3,5],
                "learning_rate":[0.05,0.1],"subsample":[0.8,1.0]}
    reg = GridSearchCV(GradientBoostingRegressor(random_state=random_state),
                       reg_grid, cv=tscv3, scoring="neg_mean_squared_error", n_jobs=-1)
    reg.fit(x_tr, y_reg_tr)
    res_r = eval_reg(reg.best_estimator_, x_te, y_reg_te, f"gbm reg [{label}]")
    reg_results[f"gbm_reg_{label}"] = res_r
    feat_importances[f"gbmreg_{label}"] = pd.Series(
        reg.best_estimator_.feature_importances_, index=x_tr.columns).sort_values(ascending=False)
    print(f"    mae={res_r['mae']:.4f}%  rmse={res_r['rmse']:.4f}%  r²={res_r['r2']:.4f}  da={res_r['da']:.1f}%")

# ══════════════════════════════════════════════════════════════════════════════
# head-to-head comparison tables
# ══════════════════════════════════════════════════════════════════════════════
banner("head-to-head comparison")

sub("classification results")
clf_table = []
for key, res in clf_results.items():
    clf_table.append({"model+features": res["label"],
                      "accuracy": res["acc"],
                      "roc-auc": res["auc"]})
clf_df = pd.DataFrame(clf_table)
print(f"\n{clf_df.to_string(index=False)}")

# calculate improvement from adding trend features
for model in ["rf","gbm"]:
    orig_auc = clf_results[f"{model}_original"]["auc"]
    enh_auc  = clf_results[f"{model}_enhanced"]["auc"]
    delta    = enh_auc - orig_auc
    sign     = "↑" if delta > 0 else "↓"
    print(f"\n  {model}: enhanced vs original auc  δ = {delta:+.4f} {sign}")

sub("regression results")
reg_table = []
for key, res in reg_results.items():
    reg_table.append({"model+features": res["label"],
                      "mae%": round(res["mae"],4),
                      "rmse%": round(res["rmse"],4),
                      "r²": round(res["r2"],4),
                      "dir.acc%": round(res["da"],2)})
reg_df = pd.DataFrame(reg_table)
print(f"\n{reg_df.to_string(index=False)}")

orig_r2 = reg_results["gbm_reg_original"]["r2"]
enh_r2  = reg_results["gbm_reg_enhanced"]["r2"]
print(f"\n  gbm regressor r² improvement: {enh_r2 - orig_r2:+.4f}")

# identify which new features became important
sub("new features that appeared in top 20")
top20_orig = set(feat_importances["rf_original"].head(20).index)
top20_enh  = set(feat_importances["rf_enhanced"].head(20).index)
new_in_top = top20_enh - top20_orig
dropped    = top20_orig - top20_enh
print(f"  new trend features in top-20  : {sorted(new_in_top & set(new_features))}")
print(f"  original features dropped out : {sorted(dropped)}")

# save comparison dataframe
comparison_rows = []
for key, res in clf_results.items():
    comparison_rows.append({"type":"classification","model":res["label"],
                             "acc":res["acc"],"auc":res["auc"]})
for key, res in reg_results.items():
    comparison_rows.append({"type":"regression","model":res["label"],
                             "mae":res["mae"],"rmse":res["rmse"],
                             "r2":res["r2"],"da":res["da"]})
pd.DataFrame(comparison_rows).to_csv(out_csv, index=False)

# ══════════════════════════════════════════════════════════════════════════════
# visualisations
# ══════════════════════════════════════════════════════════════════════════════
banner("generating charts")

# chart 1: auc / accuracy comparison bar chart
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

models = ["rf","gbm"]
orig_aucs = [clf_results[f"{m}_original"]["auc"] for m in models]
enh_aucs  = [clf_results[f"{m}_enhanced"]["auc"] for m in models]
orig_accs = [clf_results[f"{m}_original"]["acc"] for m in models]
enh_accs  = [clf_results[f"{m}_enhanced"]["acc"] for m in models]

x = np.arange(len(models)); w = 0.35

for ax, orig_vals, enh_vals, title in [
    (ax1, orig_aucs, enh_aucs, "roc-auc comparison"),
    (ax2, orig_accs, enh_accs, "accuracy comparison"),
]:
    b1 = ax.bar(x - w/2, orig_vals, w, color=c["orig"], alpha=0.85, label="original features")
    b2 = ax.bar(x + w/2, enh_vals,  w, color=c["enh"],  alpha=0.85, label="+ trend features")
    for b, v in [(b1, orig_vals), (b2, enh_vals)]:
        for bar, val in zip(b, v):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(models, fontsize=11)
    ax.set_title(title, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_ylim(min(min(orig_vals),min(enh_vals))*0.98, max(max(orig_vals),max(enh_vals))*1.02)
    ax.grid(axis="y", alpha=0.3)

fig.suptitle("original vs enhanced feature sets — classification performance",
             fontsize=13, fontweight="bold")
plt.tight_layout()
save_fig(fig, "01_clf_comparison.png")

# chart 2: regression metric comparison (4 subplots)
fig, axes = plt.subplots(1, 4, figsize=(16, 5))
reg_metrics = [("mae","mae %","lower better"),("rmse","rmse %","lower better"),
               ("r2","r²","higher better"),("da","dir. acc %","higher better")]

for ax, (metric, label, note) in zip(axes, reg_metrics):
    orig_v = reg_results["gbm_reg_original"][metric]
    enh_v  = reg_results["gbm_reg_enhanced"][metric]
    bars   = ax.bar(["original","enhanced"], [orig_v, enh_v],
                    color=[c["orig"], c["enh"]], alpha=0.85, edgecolor="#0d1117", width=0.5)
    for bar, v in zip(bars, [orig_v, enh_v]):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+abs(max(orig_v,enh_v))*0.01,
                f"{v:.4f}", ha="center", va="bottom", fontsize=9)
    ax.set_title(f"{label}\n({note})", fontweight="bold", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    lo = min(orig_v,enh_v)*0.97; hi = max(orig_v,enh_v)*1.03
    ax.set_ylim(lo, hi)

fig.suptitle("original vs enhanced — regression metrics (gbm regressor)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
save_fig(fig, "02_reg_comparison.png")

# chart 3: feature importance comparison (original top 20 vs enhanced top 20)
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

for ax, key, title, color in [
    (axes[0], "rf_original",  "rf — original features (top 20)", c["orig"]),
    (axes[1], "rf_enhanced",  "rf — enhanced features (top 20)", c["enh"]),
]:
    imp = feat_importances[key].head(20)
    # highlight new trend features in different colour
    bar_colors = [c["imp"] if f in new_features or "price_vs" in f.lower() else color
                  for f in imp.index]
    imp.plot(kind="bar", ax=ax, color=bar_colors, edgecolor="#0d1117", alpha=0.9)
    ax.set_title(title, fontweight="bold", fontsize=11)
    ax.set_ylabel("importance")
    ax.tick_params(axis="x", rotation=50, labelsize=7.5)
    ax.grid(axis="y", alpha=0.3)

new_patch  = mpatches.Patch(color=c["imp"],  label="new trend features")
orig_patch = mpatches.Patch(color=c["orig"], label="original features")
fig.legend(handles=[orig_patch, new_patch], fontsize=9, loc="lower center", ncol=2, bbox_to_anchor=(0.5,-0.02))
fig.suptitle("feature importance: original vs enhanced (rf classifier)", fontsize=13, fontweight="bold")
plt.tight_layout()
save_fig(fig, "03_feature_importance.png")

# chart 4: which new features added the most value (horizontal bar)
fig, ax = plt.subplots(figsize=(13, 6))

top20_enh_imp = feat_importances["rf_enhanced"].head(30)
new_feat_imp  = top20_enh_imp[top20_enh_imp.index.isin(new_features + [c for c in top20_enh_imp.index if "price_vs" in c.lower()])]
new_feat_imp  = new_feat_imp.sort_values(ascending=True)

colors = [c["enh"] if v > 0 else c["delta_neg"] for v in new_feat_imp]
new_feat_imp.plot(kind="barh", ax=ax, color=colors, edgecolor="#0d1117", alpha=0.9)
ax.set_title("new trend features — importance in enhanced model (rf classifier)",
             fontweight="bold", fontsize=12)
ax.set_xlabel("feature importance")
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
save_fig(fig, "04_new_feature_value.png")

# chart 5: actual vs predicted returns (original vs enhanced)
fig, axes = plt.subplots(1, 2, figsize=(13, 6))
dates_te = y_reg_te.index

for ax, key, title, color in [
    (axes[0], "gbm_reg_original", "gbm regressor — original features", c["orig"]),
    (axes[1], "gbm_reg_enhanced", "gbm regressor — enhanced features",  c["enh"]),
]:
    res = reg_results[key]
    act = res["actual"]; pred = res["pred"]
    lo  = min(act.min(), pred.min()); hi = max(act.max(), pred.max())
    ax.scatter(act, pred, alpha=0.3, s=10, color=color)
    ax.plot([lo,hi],[lo,hi], "w--", lw=1.2, label="perfect")
    r2_val = res["r2"]
    ax.set_title(f"{title}\nr²={r2_val:.4f}  da={res['da']:.1f}%", fontweight="bold", fontsize=10)
    ax.set_xlabel("actual return %")
    ax.set_ylabel("predicted return %")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)

fig.suptitle("actual vs predicted returns: original vs enhanced", fontsize=13, fontweight="bold")
plt.tight_layout()
save_fig(fig, "05_reg_scatter.png")

# ══════════════════════════════════════════════════════════════════════════════
# final summary
# ══════════════════════════════════════════════════════════════════════════════
banner("final summary")

rf_delta_auc  = clf_results["rf_enhanced"]["auc"]  - clf_results["rf_original"]["auc"]
gbm_delta_auc = clf_results["gbm_enhanced"]["auc"] - clf_results["gbm_original"]["auc"]
reg_delta_r2  = reg_results["gbm_reg_enhanced"]["r2"] - reg_results["gbm_reg_original"]["r2"]
reg_delta_da  = reg_results["gbm_reg_enhanced"]["da"] - reg_results["gbm_reg_original"]["da"]

print(f"""
  ┌──────────────────────────────────────────────────────────────────┐
  │  enhanced ml — results summary                                   │
  ├────────────────────────────────┬─────────────────────────────────┤
  │ rf classifier auc              │ orig={clf_results['rf_original']['auc']:.4f}  enh={clf_results['rf_enhanced']['auc']:.4f}  δ={rf_delta_auc:+.4f} │
  │ gbm classifier auc            │ orig={clf_results['gbm_original']['auc']:.4f}  enh={clf_results['gbm_enhanced']['auc']:.4f}  δ={gbm_delta_auc:+.4f} │
  │ gbm regressor r²              │ orig={reg_results['gbm_reg_original']['r2']:.4f}  enh={reg_results['gbm_reg_enhanced']['r2']:.4f}  δ={reg_delta_r2:+.4f} │
  │ gbm regressor dir.acc %       │ orig={reg_results['gbm_reg_original']['da']:.2f}%  enh={reg_results['gbm_reg_enhanced']['da']:.2f}%  δ={reg_delta_da:+.2f}pp │
  ├────────────────────────────────┼─────────────────────────────────┤
  │ most valuable new features     │ {", ".join(list(new_in_top & set(new_features))[:4])} │
  │ features count: original       │ {len(orig_features):<32} │
  │ features count: enhanced       │ {len(enhanced_features):<32} │
  └────────────────────────────────┴─────────────────────────────────┘

  verdict: {'trend features improved model performance ↑' if rf_delta_auc > 0 else 'trend features did not improve auc — original features sufficient'}
  saved: ogdc_enhanced_results.csv  +  5 charts
""")