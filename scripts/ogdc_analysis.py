"""
ogdc_analysis.py
================
Full statistical analysis + ML pipeline for OGDC stock price data.

Parts:
  1 — Statistical Tests (ADF, Jarque-Bera, Ljung-Box, Correlation)
  2 — Feature Engineering
  3 — Models 1-3 (Random Forest Classifier, GBM Classifier, K-Means)
  4 — Model 4 (GBM Regressor with Diebold-Mariano test)
  5 — Summary

Dependencies: pandas, numpy, scipy, scikit-learn, seaborn, matplotlib
Note: XGBoost is unavailable in this environment; scikit-learn's
      GradientBoostingClassifier / GradientBoostingRegressor are used
      as drop-in replacements (same hyperparameter names, same results quality).
"""

import sys, os, shutil, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from paths import processed, img, frontend, PATHS, ensure_dirs
ensure_dirs()

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE  = os.path.dirname(os.path.abspath(__file__)) # Scripts
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

"""
ogdc_analysis.py
================
STEP 2 — Feature engineering, ML modelling, and prediction generation.

Split:
  ┌─────────────────────────────────────────────────────────────┐
  │  Train (70%)  │  Validation (15%)  │  Test (15%)           │
  │  Jan 2020 →   │  → used for        │  → held out until     │
  │  fits model   │  hyperparam tuning │  final evaluation     │
  └─────────────────────────────────────────────────────────────┘

  - Train:      Model learns patterns from this window. 
  - Validation: GridSearchCV tunes hyperparameters ONLY here.
                Keeps test set truly unseen during model selection.
  - Test:       Touched exactly ONCE at the end. The honest score.

Inputs:   data/processed/ogdc_cleaned.csv
Outputs:  data/processed/ogdc_with_regimes.csv
          data/processed/model_predictions.csv
          outputs/images/ogdc_*.png

Run:      python scripts/ogdc_analysis.py
"""

import sys, os

# All paths are relative to the project root (one level up from scripts/)
_ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Saves processed files
def processed(filename):
    """Path inside data/processed/"""
    folder = os.path.join(_ROOT, "data", "processed")
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, filename)

# Saves images
def img_path(filename):
    """Path inside outputs/images/"""
    folder = os.path.join(_ROOT, "outputs", "images")
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, filename)

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.ensemble import (RandomForestClassifier,
                               GradientBoostingClassifier,
                               GradientBoostingRegressor)
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import (classification_report, roc_auc_score,
                              mean_absolute_error, mean_squared_error,
                              r2_score)
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# ── Config ────────────────────────────────────────────────────────────────────
CSV_PATH     = processed("ogdc_cleaned.csv")
OUTPUT_CSV   = processed("ogdc_with_regimes.csv")
PRED_CSV     = processed("model_predictions.csv")
RANDOM_STATE = 42
TRAIN_RATIO  = 0.70
VAL_RATIO    = 0.15
# Test = remaining 15%

FEATURE_COLS = [
    "Return_lag1", "Return_lag2", "Return_lag3", "Return_lag5",
    "Return_roll_mean5", "Return_roll_std5",
    "Return_roll_mean10", "Return_roll_std10",
    "RSI", "MACD", "MACD_signal", "MACD_hist",
    "DayOfWeek", "Month", "Quarter",
    "Open", "High", "Low", "Volume", "ChangeP",
]

# saves files
def save(fig, name):
    path = img_path(name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> outputs/images/{name}")

def section(title):
    print(f"\n{'='*70}\n  {title}\n{'='*70}")

# Regression
def eval_regression(actual, pred, label):
    mae  = mean_absolute_error(actual, pred) # Mean absolute error
    rmse = float(np.sqrt(mean_squared_error(actual, pred))) # root mean sq
    r2   = r2_score(actual, pred)
    da   = float(np.mean(np.sign(actual) == np.sign(pred)) * 100)
    mask = actual != 0
    mape = float(np.mean(np.abs((actual[mask]-pred[mask])/actual[mask]))*100)
    print(f"\n  [{label}]  MAE={mae:.4f}%  RMSE={rmse:.4f}%  "
          f"R2={r2:.4f}  DA={da:.2f}%  MAPE={mape:.2f}%")
    if r2 < 0:
        print(f"    WARNING: R2 negative — model worse than predicting the mean")
    return dict(mae=mae, rmse=rmse, r2=r2, da=da, mape=mape)

# ── Load ──────────────────────────────────────────────────────────────────────
section("LOADING DATA")
df = pd.read_csv(CSV_PATH, index_col="Date", parse_dates=True)

# Strip whitespace from column names (Windows CSV exports sometimes add spaces)
df.columns = df.columns.str.strip()

# Normalise column names — handles both Investing.com raw export and
# previously-cleaned CSV variants (uppercase or lowercase headers)
rename_map = {
    "Vol.":     "Volume",   "vol.":     "Volume",
    "Change %": "ChangeP",  "change %": "ChangeP",
    "Return":   "Returns",  "return":   "Returns",
    "price":    "Price",    "open":     "Open",
    "high":     "High",     "low":      "Low",
    "volume":   "Volume",   "returns":  "Returns",
    "changep":  "ChangeP",
}
df.rename(columns=rename_map, inplace=True)

if "Target"  in df.columns: df.drop(columns=["Target"], inplace=True)
df.dropna(subset=["Returns"], inplace=True)
df.sort_index(inplace=True)
print(f"  Rows: {len(df)}  |  {df.index.min().date()} to {df.index.max().date()}")

# ── Feature engineering ───────────────────────────────────────────────────────
section("FEATURE ENGINEERING")
fe = df.copy()

for lag in [1, 2, 3, 5]:
    fe[f"Return_lag{lag}"] = fe["Returns"].shift(lag)

for window in [5, 10]:
    fe[f"Return_roll_mean{window}"] = fe["Returns"].rolling(window).mean()
    fe[f"Return_roll_std{window}"]  = fe["Returns"].rolling(window).std()

def compute_rsi(series, period=14):
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

fe["RSI"]         = compute_rsi(fe["Price"], 14)
ema12             = fe["Price"].ewm(span=12, adjust=False).mean()
ema26             = fe["Price"].ewm(span=26, adjust=False).mean()
fe["MACD"]        = ema12 - ema26
fe["MACD_signal"] = fe["MACD"].ewm(span=9, adjust=False).mean()
fe["MACD_hist"]   = fe["MACD"] - fe["MACD_signal"]
fe["DayOfWeek"]   = fe.index.dayofweek
fe["Month"]       = fe.index.month
fe["Quarter"]     = fe.index.quarter
fe["Direction"]   = (fe["Returns"] > 0).astype(int)

n_before = len(fe)
fe.dropna(inplace=True)
print(f"  Dropped {n_before - len(fe)} NaN rows (warmup). Remaining: {len(fe)}")

FEATURE_COLS = [c for c in FEATURE_COLS if c in fe.columns]

# ── Train / Validation / Test split ───────────────────────────────────────────
section("TRAIN / VALIDATION / TEST SPLIT  (70 / 15 / 15)")

n       = len(fe)
n_train = int(n * TRAIN_RATIO)
n_val   = int(n * VAL_RATIO)
n_test  = n - n_train - n_val

train = fe.iloc[:n_train]
val   = fe.iloc[n_train: n_train + n_val]
test  = fe.iloc[n_train + n_val:]

print(f"\n  {'Split':<12} {'Rows':>5}   {'Start':<13} {'End':<13}  {'%':>5}")
print(f"  {'─'*55}")
for name, split in [("Train", train), ("Validation", val), ("Test", test)]:
    pct = len(split) / n * 100
    print(f"  {name:<12} {len(split):>5}   "
          f"{str(split.index.min().date()):<13} "
          f"{str(split.index.max().date()):<13}  {pct:>5.1f}%")

print(f"\n  Leakage check:")
print(f"    Train end < Val start  : {train.index.max() < val.index.min()}")
print(f"    Val end   < Test start : {val.index.max() < test.index.min()}")

X_train = train[FEATURE_COLS]; X_val = val[FEATURE_COLS]; X_test = test[FEATURE_COLS]
y_cls_train = train["Direction"].astype(int)
y_cls_val   = val["Direction"].astype(int)
y_cls_test  = test["Direction"].astype(int)
y_reg_train = train["Returns"]
y_reg_val   = val["Returns"]
y_reg_test  = test["Returns"]

# Train + val combined — used for final model refit after hyperparameter tuning
X_tv       = pd.concat([X_train, X_val])
y_cls_tv   = pd.concat([y_cls_train, y_cls_val])
y_reg_tv   = pd.concat([y_reg_train, y_reg_val])
tscv_tv    = TimeSeriesSplit(n_splits=5)   # CV within train+val
tscv_tv3   = TimeSeriesSplit(n_splits=3)

# ── Model 1: Random Forest ────────────────────────────────────────────────────
section("MODEL 1 — RANDOM FOREST CLASSIFIER")

rf = RandomForestClassifier(
    n_estimators=200, max_depth=6, random_state=RANDOM_STATE, n_jobs=-1
)
rf.fit(X_train, y_cls_train)

rf_pred_val   = rf.predict(X_val)
rf_proba_val  = rf.predict_proba(X_val)[:,1]
print(f"  [VALIDATION]  Acc={float((rf_pred_val==y_cls_val).mean()):.4f}  "
      f"AUC={roc_auc_score(y_cls_val, rf_proba_val):.4f}")

# Final evaluation on test (retrain on train+val for fairness)
rf.fit(X_tv, y_cls_tv)
rf_pred_test  = rf.predict(X_test)
rf_proba_test = rf.predict_proba(X_test)[:,1]
rf_acc_test   = float((rf_pred_test == y_cls_test).mean())
rf_auc_test   = roc_auc_score(y_cls_test, rf_proba_test)
print(f"  [TEST]        Acc={rf_acc_test:.4f}  AUC={rf_auc_test:.4f}")
print(f"\n{classification_report(y_cls_test, rf_pred_test)}")

naive_acc = float((y_cls_test == int(y_cls_train.mode()[0])).mean())
print(f"  Naive baseline: {naive_acc:.4f}   RF beats naive: {rf_acc_test > naive_acc}")

feat_imp_rf = pd.Series(rf.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(9,5))
feat_imp_rf.head(10).plot(kind="bar", ax=ax, color="steelblue", edgecolor="white")
ax.set_title("Random Forest — Top 10 Feature Importances", fontweight="bold")
ax.set_ylabel("Importance"); ax.tick_params(axis="x", rotation=35)
plt.tight_layout(); save(fig, "ogdc_rf_feature_importance.png")

# ── Model 2: GBM Classifier ───────────────────────────────────────────────────
section("MODEL 2 — GBM CLASSIFIER  (hyperparams tuned on train+val CV)")

gbm_grid = {"max_depth":[3,5], "learning_rate":[0.05,0.1], "n_estimators":[100,200]}
gbm_cv   = GridSearchCV(
    GradientBoostingClassifier(random_state=RANDOM_STATE),
    gbm_grid, cv=tscv_tv, scoring="roc_auc", n_jobs=-1
)
gbm_cv.fit(X_tv, y_cls_tv)
gbm = gbm_cv.best_estimator_
print(f"  Best params : {gbm_cv.best_params_}")
print(f"  Best CV AUC : {gbm_cv.best_score_:.4f}")

# Val score (train-only model)
gbm_val_model = GradientBoostingClassifier(
    random_state=RANDOM_STATE, **gbm_cv.best_params_
)
gbm_val_model.fit(X_train, y_cls_train)
gbm_pred_val  = gbm_val_model.predict(X_val)
gbm_proba_val = gbm_val_model.predict_proba(X_val)[:,1]
print(f"  [VALIDATION]  Acc={float((gbm_pred_val==y_cls_val).mean()):.4f}  "
      f"AUC={roc_auc_score(y_cls_val, gbm_proba_val):.4f}")

gbm_pred_test  = gbm.predict(X_test)
gbm_proba_test = gbm.predict_proba(X_test)[:,1]
gbm_acc_test   = float((gbm_pred_test == y_cls_test).mean())
gbm_auc_test   = roc_auc_score(y_cls_test, gbm_proba_test)
print(f"  [TEST]        Acc={gbm_acc_test:.4f}  AUC={gbm_auc_test:.4f}")
print(f"\n{classification_report(y_cls_test, gbm_pred_test)}")

# ── Model 3: K-Means ──────────────────────────────────────────────────────────
section("MODEL 3 — K-MEANS REGIME DETECTION  (unsupervised, no split needed)")

regime_data = fe[["Returns","Volume","RSI"]].dropna()
X_scaled    = StandardScaler().fit_transform(regime_data)

sil_scores = {}; inertias = []
for k in range(2,9):
    km  = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    lbl = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    sil_scores[k] = silhouette_score(X_scaled, lbl)
    print(f"  k={k}: inertia={km.inertia_:.1f}  silhouette={sil_scores[k]:.4f}")

optimal_k    = max(sil_scores, key=sil_scores.get)
km_final     = KMeans(n_clusters=optimal_k, random_state=RANDOM_STATE, n_init=10)
regime_lbl   = km_final.fit_predict(X_scaled)
regime_data  = regime_data.copy(); regime_data["Regime"] = regime_lbl
regime_summary  = regime_data.groupby("Regime")["Returns"].agg(["mean","std","count"])
print(f"\n  Optimal K={optimal_k}  (Silhouette={sil_scores[optimal_k]:.4f})")
print(regime_summary.round(4).to_string())

fe["Regime"] = np.nan
fe.loc[regime_data.index, "Regime"] = regime_lbl.astype(float)

fig, (ax1,ax2) = plt.subplots(1,2,figsize=(12,4))
ax1.plot(list(range(2,9)), inertias, "bo-", lw=2, markersize=8)
ax1.set_xlabel("k"); ax1.set_ylabel("Inertia"); ax1.set_title("Elbow", fontweight="bold"); ax1.grid(alpha=0.3)
ax2.plot(list(sil_scores.keys()), list(sil_scores.values()), "rs-", lw=2, markersize=8)
ax2.set_xlabel("k"); ax2.set_ylabel("Silhouette"); ax2.set_title("Silhouette", fontweight="bold"); ax2.grid(alpha=0.3)
plt.suptitle("K-Means Optimal K", fontweight="bold"); plt.tight_layout()
save(fig, "ogdc_kmeans_elbow.png")

colors = plt.cm.tab10.colors
fig, ax = plt.subplots(figsize=(14,5))
ax.plot(fe.index, fe["Price"], color="lightgrey", lw=1, zorder=1)
for r in range(optimal_k):
    mask = fe["Regime"] == r
    avg  = regime_summary.loc[r,"mean"]
    ax.scatter(fe.index[mask], fe["Price"].values[mask], color=colors[r],
               s=8, zorder=2, alpha=0.7, label=f"Regime {r} (avg={avg:+.2f}%)")
ax.set_title("Price Coloured by Market Regime", fontweight="bold")
ax.set_ylabel("Price (PKR)"); ax.legend(fontsize=8); ax.grid(alpha=0.2)
plt.tight_layout(); save(fig, "ogdc_regimes.png")

fig, ax = plt.subplots(figsize=(6,5))
sns.heatmap(df[["Price","Volume","Returns"]].corr(), annot=True, fmt=".3f",
            cmap="RdYlGn", center=0, ax=ax, linewidths=0.5)
ax.set_title("Correlation Matrix", fontweight="bold")
plt.tight_layout(); save(fig, "ogdc_correlation_heatmap.png")

# ── Model 4: GBM Regressor ────────────────────────────────────────────────────
section("MODEL 4 — GBM REGRESSOR  (hyperparams tuned on train+val CV)")

reg_grid = {"n_estimators":[100,200], "max_depth":[3,5],
            "learning_rate":[0.05,0.1], "subsample":[0.8,1.0]}
gbm_reg_cv = GridSearchCV(
    GradientBoostingRegressor(random_state=RANDOM_STATE),
    reg_grid, cv=tscv_tv3, scoring="neg_mean_squared_error", n_jobs=-1
)
gbm_reg_cv.fit(X_tv, y_reg_tv)
gbm_reg = gbm_reg_cv.best_estimator_
print(f"  Best params: {gbm_reg_cv.best_params_}")

# Val score (train-only model)
gbm_reg_val = GradientBoostingRegressor(random_state=RANDOM_STATE, **gbm_reg_cv.best_params_)
gbm_reg_val.fit(X_train, y_reg_train)
reg_pred_val = gbm_reg_val.predict(X_val)
eval_regression(y_reg_val.values, reg_pred_val, "VALIDATION")

# Test score
reg_pred_test = gbm_reg.predict(X_test)
actual_test   = y_reg_test.values
reg_m = eval_regression(actual_test, reg_pred_test, "TEST")

naive_mae = mean_absolute_error(actual_test, np.zeros_like(actual_test))
print(f"\n  Naive-zero MAE: {naive_mae:.4f}%   Model beats naive: {reg_m['mae'] < naive_mae}")

e1 = actual_test - reg_pred_test
e2 = actual_test
d  = e1**2 - e2**2
dm = d.mean() / np.sqrt(max(np.var(d, ddof=1)/len(d), 1e-12))
dm_p = 2*(1 - stats.norm.cdf(abs(dm)))
print(f"  Diebold-Mariano: DM={dm:.4f}  p={dm_p:.4f}  "
      f"({'significant' if dm_p < 0.05 else 'not significant'})")

residuals = actual_test - reg_pred_test

fig, ax = plt.subplots(figsize=(6,6))
ax.scatter(actual_test, reg_pred_test, alpha=0.4, s=14, color="steelblue")
lo,hi = min(actual_test.min(),reg_pred_test.min()), max(actual_test.max(),reg_pred_test.max())
ax.plot([lo,hi],[lo,hi], "r--", lw=1.5)
ax.set_xlabel("Actual Return (%)"); ax.set_ylabel("Predicted Return (%)")
ax.set_title(f"Actual vs Predicted — Test  (R²={reg_m['r2']:.4f})", fontweight="bold")
ax.grid(alpha=0.25); plt.tight_layout(); save(fig, "ogdc_reg_actual_vs_pred.png")

fig, (ax1,ax2) = plt.subplots(1,2,figsize=(12,4))
ax1.hist(residuals, bins=40, color="mediumpurple", edgecolor="white", alpha=0.85)
ax1.axvline(0, color="red", linestyle="--"); ax1.set_title("Residuals Distribution", fontweight="bold")
(osm,osr),(slope,intercept,_) = stats.probplot(residuals, dist="norm")
ax2.scatter(osm, osr, s=8, alpha=0.5, color="mediumpurple")
ax2.plot(osm, slope*np.array(osm)+intercept, "r-", lw=1.5)
ax2.set_title("Q-Q Plot", fontweight="bold"); ax2.grid(alpha=0.3)
plt.tight_layout(); save(fig, "ogdc_reg_residuals.png")

fig, ax = plt.subplots(figsize=(14,4))
ax.plot(test.index, actual_test, label="Actual", lw=1.2, color="steelblue")
ax.plot(test.index, reg_pred_test, label="Predicted", lw=1, color="tomato", linestyle="--", alpha=0.8)
ax.axhline(0, color="black", lw=0.7, alpha=0.4)
ax.set_title("Actual vs Predicted Returns — Test Period", fontweight="bold")
ax.legend(); ax.grid(alpha=0.2); plt.tight_layout(); save(fig, "ogdc_reg_timeseries.png")

fig, ax = plt.subplots(figsize=(14,4))
ax.plot(test.index, residuals, color="darkorange", lw=0.9, alpha=0.8)
ax.axhline(0, color="red", linestyle="--", lw=1)
ax.fill_between(test.index, residuals, 0, alpha=0.2, color="darkorange")
ax.set_title("Residuals Over Time", fontweight="bold"); ax.grid(alpha=0.2)
plt.tight_layout(); save(fig, "ogdc_reg_residuals_time.png")

feat_imp_reg = pd.Series(gbm_reg.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
fig, (ax1,ax2) = plt.subplots(1,2,figsize=(15,5))
feat_imp_reg.head(15).plot(kind="bar", ax=ax1, color="tomato", edgecolor="white")
ax1.set_title("GBM Regressor — Top 15", fontweight="bold"); ax1.tick_params(axis="x", rotation=40)
feat_imp_rf.head(15).plot(kind="bar", ax=ax2, color="steelblue", edgecolor="white")
ax2.set_title("RF Classifier — Top 15", fontweight="bold"); ax2.tick_params(axis="x", rotation=40)
plt.suptitle("Feature Importance Comparison", fontweight="bold"); plt.tight_layout()
save(fig, "ogdc_feature_importance_comparison.png")

# ── Save predictions ──────────────────────────────────────────────────────────
section("SAVING PREDICTIONS")

val_preds = pd.DataFrame({
    "Date":                  val.index,
    "split":                 "validation",
    "actual_return":         y_reg_val.values,
    "actual_direction":      y_cls_val.values,
    "rf_direction":          rf_pred_val,
    "rf_confidence":         rf_proba_val,
    "gbm_direction":         gbm_pred_val,
    "gbm_confidence":        gbm_proba_val,
    "reg_predicted_return":  reg_pred_val,
    "actual_price":          val["Price"].values,
})
test_preds = pd.DataFrame({
    "Date":                  test.index,
    "split":                 "test",
    "actual_return":         actual_test,
    "actual_direction":      y_cls_test.values,
    "rf_direction":          rf_pred_test,
    "rf_confidence":         rf_proba_test,
    "gbm_direction":         gbm_pred_test,
    "gbm_confidence":        gbm_proba_test,
    "reg_predicted_return":  reg_pred_test,
    "actual_price":          test["Price"].values,
})

all_preds = pd.concat([val_preds, test_preds], ignore_index=True)
all_preds.to_csv(PRED_CSV, index=False)
print(f"  Saved model_predictions.csv  ({len(all_preds)} rows: {len(val_preds)} val + {len(test_preds)} test)")

# Backward-compatible format for ogdc_contradiction_analysis.py
compat = test_preds.copy()
compat["model_predicted_direction"] = compat["rf_direction"].map({1:"Up",0:"Down"})
compat["model_confidence"]          = compat["rf_confidence"]
compat["model_predicted_return_pct"]= compat["reg_predicted_return"]
compat[["Date","model_predicted_direction","model_confidence",
         "model_predicted_return_pct","actual_return","actual_price"]].to_csv(
    processed("model_predictions_compat.csv"), index=False
)

fe.to_csv(OUTPUT_CSV)
print(f"  Saved ogdc_with_regimes.csv  ({len(fe)} rows)")

# ── Summary ───────────────────────────────────────────────────────────────────
section("SUMMARY")
print(f"""
  Split       Rows    Start         End
  Train       {len(train):<6}  {train.index.min().date()}  {train.index.max().date()}
  Validation  {len(val):<6}  {val.index.min().date()}  {val.index.max().date()}
  Test        {len(test):<6}  {test.index.min().date()}  {test.index.max().date()}

  MODEL RESULTS (TEST SET)
  RF  Classifier:  Accuracy={rf_acc_test:.4f}  AUC={rf_auc_test:.4f}
  GBM Classifier:  Accuracy={gbm_acc_test:.4f}  AUC={gbm_auc_test:.4f}
  GBM Regressor:   R2={reg_m['r2']:.4f}  DA={reg_m['da']:.1f}%  MAE={reg_m['mae']:.4f}%
  K-Means:         K={optimal_k}  Silhouette={sil_scores[optimal_k]:.4f}
""")


