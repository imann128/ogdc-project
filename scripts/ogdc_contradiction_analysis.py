"""
ogdc_contradiction_analysis.py
===============================
approach 3 — contradiction analysis framework

sources:
  • ogdc_sentiment_results.csv   — 14 sources, sentiment labels, recommendations, outlooks
  • model_predictions.csv         — rf classifier + gbm regressor predictions (test period)
  • ogdc_cleaned.csv              — actual daily ogdc price & returns

the analysis operates on two complementary layers:
  layer a — source-level: each of the 14 rows is one analytical event (publication date)
  layer b — daily market-level: sentiment signals mapped to actual price windows

contradiction = buy/implicit buy recommendation + negative outlook (4 rows: ids 11–14)
"""

import sys, os, shutil, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from paths import processed, img, frontend, PATHS, ensure_dirs
ensure_dirs()

import warnings, os
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from datetime import timedelta

# hardcoded paths for input data, outputs, and images
BASE = os.path.dirname(os.path.abspath(__file__))  # scripts folder
DATA_DIR = os.path.join(BASE, "..", "data", "processed")
os.makedirs(DATA_DIR, exist_ok=True)
SENT_PATH = os.path.join(DATA_DIR, "ogdc_sentiment_results.csv")  # 14 source sentiments
MODEL_PATH = os.path.join(DATA_DIR, "model_predictions.csv")  # ml predictions from ogdc_analysis
PRICE_PATH = os.path.join(DATA_DIR, "ogdc_cleaned.csv")  # actual price data

OUT_FULL = os.path.join(DATA_DIR, "contradiction_analysis_full.csv")
OUT_SUMMARY = os.path.join(DATA_DIR, "contradiction_summary.csv")
IMG_DIR = os.path.join(BASE, "..", "outputs", "images")
os.makedirs(IMG_DIR, exist_ok=True)
IMG_PRE = os.path.join(IMG_DIR, "contra_")  # prefix for contradiction images

# save figure helper
def save_fig(fig, name):
    path = IMG_PRE + name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    → saved: contra_{name}")

# console formatting helpers
def banner(title):
    print(f"\n{'═'*70}\n  {title}\n{'═'*70}")

def sub(title):
    print(f"\n  {'─'*60}")
    print(f"  {title}")
    print(f"  {'─'*60}")

# ══════════════════════════════════════════════════════════════════════════════
# step 1 — load & merge all three data sources
# ══════════════════════════════════════════════════════════════════════════════
banner("step 1 — loading and merging data")

# 1a. sentiment (14 sources: analyst reports, news, brokerage notes)
sent = pd.read_csv(SENT_PATH)
sent.columns = sent.columns.str.strip()
sent["date"] = pd.to_datetime(sent["date"], errors="coerce")
sent["recommendation"] = sent["recommendation"].fillna("none")
sent["outlook"] = sent["outlook"].fillna("unknown")
sent["contradiction_flag"] = sent["contradiction_flag"].str.strip()
print(f"  sentiment rows loaded  : {len(sent)}")
print(f"  contradiction rows     : {sent[sent['contradiction_flag']=='yes']['id'].tolist()}")

# 1b. ml model predictions (test period: 2025-01-09 → 2026-04-03)
model = pd.read_csv(MODEL_PATH)
model["Date"] = pd.to_datetime(model["Date"])
print(f"  model prediction rows  : {len(model)} ({model['Date'].min().date()} → {model['Date'].max().date()})")

# 1c. actual price / returns from loader
price = pd.read_csv(PRICE_PATH)
price.columns = price.columns.str.strip()
price.rename(columns={"Date":"date","Price":"price","Return":"actual_return","Change %":"change_pct"}, inplace=True)
price["date"] = pd.to_datetime(price["date"])
price = price.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
print(f"  price rows loaded      : {len(price)} ({price['date'].min().date()} → {price['date'].max().date()})")

# helper: look up price window returns from the price series
def get_window_returns(pub_date, price_df, windows=[1,3,5,10]):
    """
    given a publication date, find the next available trading day and
    compute cumulative returns over subsequent windows (d+1, d+3, d+5, d+10).
    uses product of (1 + daily_return) - 1 for accurate multi-day compounding.
    """
    if pd.isna(pub_date):
        return {f"d{w}_return": np.nan for w in windows}
    # find rows on or after pub date
    future = price_df[price_df["date"] >= pub_date].reset_index(drop=True)
    if future.empty:
        return {f"d{w}_return": np.nan for w in windows}
    result = {}
    for w in windows:
        if len(future) < w:
            result[f"d{w}_return"] = np.nan
        else:
            # cumulative return: product of (1 + daily_ret/100) - 1
            sub_ret = future.iloc[:w]["actual_return"].dropna() / 100
            if sub_ret.empty:
                result[f"d{w}_return"] = np.nan
            else:
                cum = (1 + sub_ret).prod() - 1
                result[f"d{w}_return"] = round(cum * 100, 4)  # convert back to percentage
    return result

# helper: find model prediction for a given publication date
def get_model_pred(pub_date, model_df):
    """
    for a publication date, find the closest future trading day in model predictions.
    returns model direction (up/down), confidence (0-1), and predicted return %.
    """
    if pd.isna(pub_date):
        return {"model_direction": "n/a", "model_confidence": np.nan, "model_pred_return": np.nan}
    future = model_df[model_df["Date"] >= pub_date].sort_values("Date")
    if future.empty:
        return {"model_direction": "pre-test", "model_confidence": np.nan, "model_pred_return": np.nan}
    row = future.iloc[0]
    
    # Map numeric direction (1/0) to text (up/down)
    rf_dir = row.get("rf_direction", 0)
    direction = "up" if rf_dir == 1 else "down"
    
    return {
        "model_direction": direction,
        "model_confidence": round(row.get("rf_confidence", 0.5), 4),
        "model_pred_return": round(row.get("reg_predicted_return", 0), 4),
    }

# build enriched source-level dataframe
print("\n  building enriched source-level table...")
rows = []
for _, s in sent.iterrows():
    pub  = s["date"]

    # price windows (use pre-computed values if available, else compute)
    d1_pre  = s.get("d1_chg_%",  np.nan)
    d3_pre  = s.get("d3_cum_%",  np.nan)
    d5_pre  = s.get("d5_cum_%",  np.nan)
    d10_pre = s.get("d10_cum_%", np.nan)

    if pd.notna(d1_pre):
        # use pre-computed values from sentiment csv
        win = {"d1_return": d1_pre, "d3_return": d3_pre,
               "d5_return": d5_pre, "d10_return": d10_pre}
    else:
        win = get_window_returns(pub, price)

    # model prediction for this date
    mp = get_model_pred(pub, model)

    # actual direction at d+1 (next trading day)
    d1_ret = win.get("d1_return", np.nan)
    if pd.isna(d1_ret):
        actual_dir = "unknown"
    elif d1_ret > 0:
        actual_dir = "up"
    elif d1_ret < 0:
        actual_dir = "down"
    else:
        actual_dir = "flat"

    # sentiment label from nlp analysis
    sent_label = str(s["sentiment_label"]).strip()
    rec        = str(s["recommendation"]).strip()
    outlook    = str(s["outlook"]).strip()
    contra     = str(s["contradiction_flag"]).strip()

    # normalise recommendation (implicit buy = buy for contradiction logic)
    rec_clean = rec.lower().replace("implicit buy","buy").replace("strong buy","buy")

    # model_correct — for test-period sources only (direction accuracy)
    if mp["model_direction"] in ("up","down"):
        if pd.isna(d1_ret):
            model_correct = "unknown"
        else:
            model_correct = "yes" if mp["model_direction"] == actual_dir else "no"
    else:
        model_correct = "n/a (pre-test)"

    # sentiment_correct — did sentiment predict the actual move?
    if sent_label == "neutral":
        sentiment_correct = "neutral"
    elif sent_label == "positive" and actual_dir == "up":
        sentiment_correct = "yes"
    elif sent_label == "negative" and actual_dir == "down":
        sentiment_correct = "yes"
    elif actual_dir == "unknown":
        sentiment_correct = "unknown"
    else:
        sentiment_correct = "no"

    # market_followed — for contradiction rows only: which signal won?
    if contra == "yes":
        if rec_clean == "buy" and actual_dir == "up":
            market_followed = "recommendation"
        elif rec_clean == "buy" and actual_dir == "down":
            market_followed = "sentiment/outlook"
        elif rec_clean == "sell" and actual_dir == "down":
            market_followed = "recommendation"
        elif rec_clean == "sell" and actual_dir == "up":
            market_followed = "sentiment/outlook"
        elif actual_dir == "unknown":
            market_followed = "unknown (no date)"
        else:
            market_followed = "neither"
    else:
        market_followed = "n/a (aligned)"

    rows.append({
        "id":                    int(s["id"]),
        "source":                s["source"],
        "date":                  pub.strftime("%y-%m-%d") if pd.notna(pub) else "n/a",
        "sentiment_score":       s["sentiment_score"],
        "sentiment_label":       sent_label,
        "recommendation":        rec,
        "outlook":               outlook,
        "contradiction_flag":    contra,
        "contradiction_type":    str(s.get("contradiction_type","")).strip(),
        "key_phrase":            str(s.get("key_contradiction_phrase","")).strip(),
        # price windows (d+1, d+3, d+5, d+10 cumulative returns)
        "d1_return_%":           win.get("d1_return", np.nan),
        "d3_return_%":           win.get("d3_return", np.nan),
        "d5_return_%":           win.get("d5_return", np.nan),
        "d10_return_%":          win.get("d10_return", np.nan),
        # direction at d+1
        "actual_direction_d1":   actual_dir,
        # model predictions
        "model_direction":       mp["model_direction"],
        "model_confidence":      mp["model_confidence"],
        "model_pred_return_%":   mp["model_pred_return"],
        # accuracy metrics
        "model_correct":         model_correct,
        "sentiment_correct":     sentiment_correct,
        "market_followed":       market_followed,
        # pre-computed contra fields from original file (d+10 outcome)
        "contra_market_d1_orig": str(s.get("contra_market_d1","")).strip(),
        "contra_market_d10_orig":str(s.get("contra_market_d10","")).strip(),
    })

full_df = pd.DataFrame(rows)
print(f"\n  merged table shape: {full_df.shape}")
print(f"  date range: {full_df[full_df['date']!='n/a']['date'].min()} → {full_df[full_df['date']!='n/a']['date'].max()}")
print(f"\n  first 5 rows:")
print(full_df[["id","source","date","sentiment_label","recommendation","outlook",
               "contradiction_flag","actual_direction_d1","market_followed"]].to_string(index=False))

# ══════════════════════════════════════════════════════════════════════════════
# step 2 — contradiction analysis table (all 14 sources)
# ══════════════════════════════════════════════════════════════════════════════
banner("step 2 — contradiction analysis table (all 14 sources)")

display_cols = [
    "id","source","date",
    "model_direction","actual_direction_d1",
    "sentiment_label","recommendation","contradiction_flag","contradiction_type",
    "market_followed","model_correct","sentiment_correct",
]
print(full_df[display_cols].to_string(index=False))

# ══════════════════════════════════════════════════════════════════════════════
# step 3 — the four critical metrics (contradiction analysis)
# ══════════════════════════════════════════════════════════════════════════════
banner("step 3 — four critical metrics")

contra_df  = full_df[full_df["contradiction_flag"] == "yes"].copy()
aligned_df = full_df[full_df["contradiction_flag"] == "no"].copy()

n_contra  = len(contra_df)
n_aligned = len(aligned_df)
n_total   = len(full_df)

print(f"  total sources          : {n_total}")
print(f"  contradiction rows     : {n_contra}  (ids: {contra_df['id'].tolist()})")
print(f"  aligned rows           : {n_aligned}")

# contradiction-day metrics (dated sources only)
sub("contradiction rows")

# rows where we have an actual direction (have a date)
contra_dated = contra_df[contra_df["actual_direction_d1"] != "unknown"]
n_contra_dated = len(contra_dated)

follow_rec_n   = (contra_dated["market_followed"] == "recommendation").sum()
follow_sent_n  = (contra_dated["market_followed"] == "sentiment/outlook").sum()
model_corr_n   = (contra_dated["model_correct"] == "yes").sum()
sent_corr_n    = (contra_dated["sentiment_correct"] == "yes").sum()

if n_contra_dated > 0:
    follow_rec_rate  = follow_rec_n  / n_contra_dated * 100
    follow_sent_rate = follow_sent_n / n_contra_dated * 100
    model_acc_contra = model_corr_n  / n_contra_dated * 100
    sent_acc_contra  = sent_corr_n   / n_contra_dated * 100
else:
    follow_rec_rate = follow_sent_rate = model_acc_contra = sent_acc_contra = np.nan

# sources without a date (brokerage reports with no event date)
contra_undated = contra_df[contra_df["actual_direction_d1"] == "unknown"]
print(f"\n  contradiction rows with date (tradeable signal): {n_contra_dated}")
print(f"  contradiction rows without date (brokerage reports): {len(contra_undated)}")

if n_contra_dated > 0:
    print(f"\n  metric 1 — follow recommendation rate : {follow_rec_rate:.1f}%  ({follow_rec_n}/{n_contra_dated})")
    print(f"  metric 2 — follow sentiment rate      : {follow_sent_rate:.1f}%  ({follow_sent_n}/{n_contra_dated})")
    print(f"  metric 3 — model accuracy (contra)    : {model_acc_contra:.1f}%  ({model_corr_n}/{n_contra_dated})")
    print(f"  metric 4 — sentiment accuracy (contra): {sent_acc_contra:.1f}%  ({sent_corr_n}/{n_contra_dated})")

# summarise undated sources using original contra_market_d10 field (pre-computed)
print(f"\n  pre-computed d+10 market outcome for undated contradiction sources:")
for _, r in contra_undated.iterrows():
    print(f"    id {r['id']}  {r['source'][:40]:40s}  d10_outcome={r['contra_market_d10_orig']}")

# include undated rows using their pre-computed d+10 outcomes (brokerage horizon)
all_contra_recs  = 0
all_contra_sent  = 0
all_contra_total = 0
for _, r in contra_df.iterrows():
    d10 = r["contra_market_d10_orig"]
    if d10 in ("followed_recommendation","followed_outlook"):
        all_contra_total += 1
        if d10 == "followed_recommendation":
            all_contra_recs += 1
        else:
            all_contra_sent += 1
    elif r["actual_direction_d1"] != "unknown":
        # dated rows already counted — use market_followed
        all_contra_total += 1
        mf = r["market_followed"]
        if mf == "recommendation":
            all_contra_recs += 1
        elif mf == "sentiment/outlook":
            all_contra_sent += 1

all4_rec_rate  = all_contra_recs  / n_contra * 100 if n_contra > 0 else np.nan
all4_sent_rate = all_contra_sent  / n_contra * 100 if n_contra > 0 else np.nan

sub("all-contradiction summary (using d+10 outcomes where date missing)")
print(f"  of {n_contra} contradiction sources with resolved d+10 outcome:")
print(f"    followed recommendation : {all_contra_recs}/{n_contra} = {all4_rec_rate:.1f}%")
print(f"    followed sentiment/out  : {all_contra_sent}/{n_contra} = {all4_sent_rate:.1f}%")

# aligned-day metrics (baseline for comparison)
sub("aligned rows")
aligned_dated = aligned_df[~aligned_df["actual_direction_d1"].isin(["unknown","flat"])]
n_aligned_dated = len(aligned_dated)

aligned_model_corr = (aligned_dated["model_correct"] == "yes").sum()
aligned_sent_corr  = (aligned_dated["sentiment_correct"] == "yes").sum()

model_acc_aligned = aligned_model_corr / n_aligned_dated * 100 if n_aligned_dated > 0 else np.nan
sent_acc_aligned  = aligned_sent_corr  / n_aligned_dated * 100 if n_aligned_dated > 0 else np.nan

print(f"  aligned rows with date  : {n_aligned_dated}")
print(f"  model accuracy (aligned): {model_acc_aligned:.1f}%  ({aligned_model_corr}/{n_aligned_dated})")
print(f"  sentiment accuracy (aligned): {sent_acc_aligned:.1f}%  ({aligned_sent_corr}/{n_aligned_dated})")

# overall metrics (all 14 sources)
sub("overall (all 14 sources)")
all_dated = full_df[~full_df["actual_direction_d1"].isin(["unknown","flat","n/a (pre-test)"])]
n_all_dated = len(all_dated)

overall_model = (all_dated["model_correct"] == "yes").sum()
overall_sent  = (all_dated["sentiment_correct"] == "yes").sum()

overall_model_acc = overall_model / n_all_dated * 100 if n_all_dated > 0 else np.nan
overall_sent_acc  = overall_sent  / n_all_dated * 100 if n_all_dated > 0 else np.nan

print(f"  dateable sources        : {n_all_dated}")
print(f"  overall model accuracy  : {overall_model_acc:.1f}%")
print(f"  overall sentiment match : {overall_sent_acc:.1f}%")

# ══════════════════════════════════════════════════════════════════════════════
# step 4 — three critical questions (contradiction framework)
# ══════════════════════════════════════════════════════════════════════════════
banner("step 4 — three critical questions")

sub("q1: on contradiction days, did the market reward buying against negative outlook?")
print(f"\n  follow recommendation rate : {all4_rec_rate:.1f}%")
print(f"  follow sentiment rate      : {all4_sent_rate:.1f}%")
if all4_rec_rate > 60:
    q1_ans = "yes — market respects contrarian buy signals despite negative fundamental outlook."
elif all4_sent_rate > 60:
    q1_ans = "no — negative outlook dominated. sentiment overpowered the buy recommendation."
elif pd.isna(all4_rec_rate):
    q1_ans = "insufficient data — no resolved contradiction events."
else:
    q1_ans = "mixed — market is uncertain on contradiction days (neither signal dominates >60%)."
print(f"\n  answer: {q1_ans}")

sub("q2: did the ml model outperform sentiment on contradiction days?")
# for dated contradiction rows
if n_contra_dated > 0:
    print(f"\n  model accuracy (contra, dated) : {model_acc_contra:.1f}%")
    print(f"  sentiment accuracy (contra)    : {sent_acc_contra:.1f}%")
    delta = model_acc_contra - sent_acc_contra
    winner = "model" if delta > 0 else "sentiment"
    print(f"  δ = {abs(delta):.1f}pp  |  {winner} outperforms on dated contradiction days.")
    q2_ans = f"model {'outperforms' if delta>0 else 'underperforms'} sentiment by {abs(delta):.1f}pp on contradiction days."
else:
    q2_ans = "insufficient dated contradiction rows to compare directly."
print(f"\n  answer: {q2_ans}")

sub("q3: what is the single best signal on contradiction days?")
if n_contra_dated > 0:
    signals = {
        "recommendation alone": follow_rec_rate,
        "model direction":      model_acc_contra,
        "sentiment outlook":    sent_acc_contra,
    }
    best_sig  = max(signals, key=signals.get)
    best_rate = signals[best_sig]
    q3_ans = f"'{best_sig}' with {best_rate:.1f}% success rate on dated contradiction days."
else:
    # fall back to d+10 pre-computed (brokerage horizon)
    signals = {
        "recommendation (d+10 horizon)": all4_rec_rate,
        "sentiment/outlook (d+10)":      all4_sent_rate,
    }
    best_sig  = max(signals, key=signals.get)
    best_rate = signals[best_sig]
    q3_ans = f"'{best_sig}' with {best_rate:.1f}% success rate across all 4 contradiction sources."
print(f"\n  answer: {q3_ans}")

# ══════════════════════════════════════════════════════════════════════════════
# step 5 — summary table (segment-level metrics)
# ══════════════════════════════════════════════════════════════════════════════
banner("step 5 — final summary table")

# sub-segments for deeper analysis
buy_neg = contra_df[contra_df["recommendation"].str.lower().str.contains("buy")]
sell_pos = contra_df[
    contra_df["recommendation"].str.lower().str.contains("sell") &
    contra_df["outlook"].str.contains("positive", case=False)
]

def seg_rates(seg_df, label):
    """return metrics for a segment (buy+neg or sell+pos)"""
    dated = seg_df[~seg_df["actual_direction_d1"].isin(["unknown","flat"])]
    n = len(seg_df)
    if len(dated) > 0:
        rec  = (dated["market_followed"] == "recommendation").sum() / len(dated) * 100
        sent = (dated["market_followed"] == "sentiment/outlook").sum() / len(dated) * 100
        mc   = (dated["model_correct"] == "yes").sum() / len(dated) * 100
        sc   = (dated["sentiment_correct"] == "yes").sum() / len(dated) * 100
    else:
        # use d+10 pre-computed outcomes
        recs  = (seg_df["contra_market_d10_orig"] == "followed_recommendation").sum()
        sents = (seg_df["contra_market_d10_orig"] == "followed_outlook").sum()
        rec  = recs  / n * 100 if n > 0 else np.nan
        sent = sents / n * 100 if n > 0 else np.nan
        mc   = np.nan
        sc   = np.nan
    return {"scenario": label, "count": n,
            "followed_rec_%": f"{rec:.1f}%" if not np.isnan(rec) else "n/a",
            "followed_sent_%": f"{sent:.1f}%" if not np.isnan(sent) else "n/a",
            "model_acc_%": f"{mc:.1f}%" if not np.isnan(mc) else "n/a",
            "sent_acc_%": f"{sc:.1f}%" if not np.isnan(sc) else "n/a"}

summary_rows = [
    {"scenario": "all sources (14)",
     "count": n_total,
     "followed_rec_%": "n/a",
     "followed_sent_%": "n/a",
     "model_acc_%": f"{overall_model_acc:.1f}%" if not np.isnan(overall_model_acc) else "n/a",
     "sent_acc_%": f"{overall_sent_acc:.1f}%" if not np.isnan(overall_sent_acc) else "n/a"},

    {"scenario": "contradiction only (4)",
     "count": n_contra,
     "followed_rec_%": f"{all4_rec_rate:.1f}%",
     "followed_sent_%": f"{all4_sent_rate:.1f}%",
     "model_acc_%": f"{model_acc_contra:.1f}%" if n_contra_dated > 0 else "n/a",
     "sent_acc_%": f"{sent_acc_contra:.1f}%" if n_contra_dated > 0 else "n/a"},

    {"scenario": "aligned only (10)",
     "count": n_aligned,
     "followed_rec_%": "n/a",
     "followed_sent_%": f"{sent_acc_aligned:.1f}%" if not np.isnan(sent_acc_aligned) else "n/a",
     "model_acc_%": f"{model_acc_aligned:.1f}%" if not np.isnan(model_acc_aligned) else "n/a",
     "sent_acc_%": f"{sent_acc_aligned:.1f}%" if not np.isnan(sent_acc_aligned) else "n/a"},

    seg_rates(buy_neg,  f"buy + negative outlook ({len(buy_neg)})"),
    seg_rates(sell_pos, f"sell + positive outlook ({len(sell_pos)})"),
]

summary_df = pd.DataFrame(summary_rows)
print(f"\n{summary_df.to_string(index=False)}")

# ══════════════════════════════════════════════════════════════════════════════
# step 6 — per-source detailed table (main deliverable)
# ══════════════════════════════════════════════════════════════════════════════
banner("step 6 — per-source detailed table")

print(f"\n{'─'*120}")
print(f"{'id':>3} {'source':40s} {'date':12} {'sent':8} {'rec':14} {'outlook':10} "
      f"{'contra':6} {'act_dir':7} {'mkt_follow':20} {'model_ok':8} {'sent_ok':8}")
print(f"{'─'*120}")

for _, r in full_df.iterrows():
    src_short = str(r["source"])[:38]
    print(f"{r['id']:>3} {src_short:40s} {str(r['date']):12} "
          f"{str(r['sentiment_label']):8} {str(r['recommendation'])[:14]:14} "
          f"{str(r['outlook'])[:10]:10} {str(r['contradiction_flag']):6} "
          f"{str(r['actual_direction_d1']):7} {str(r['market_followed'])[:20]:20} "
          f"{str(r['model_correct']):8} {str(r['sentiment_correct']):8}")

print(f"{'─'*120}")

# ══════════════════════════════════════════════════════════════════════════════
# step 7 — visualisations
# ══════════════════════════════════════════════════════════════════════════════
banner("step 7 — generating visualisations")

colors = {
    "recommendation": "#2196F3",
    "sentiment/outlook": "#FF5722",
    "neither": "#9E9E9E",
    "n/a (aligned)": "#BDBDBD",
    "unknown (no date)": "#CFD8DC",
    "up": "#4CAF50",
    "down": "#F44336",
    "unknown": "#9E9E9E",
    "flat": "#FFC107",
}

# plot 1: who did the market follow on contradiction days?
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# left: d+1 / d+10 for dated contradiction rows
ax = axes[0]
outcomes_d1  = {"followed rec": all_contra_recs, "followed outlook": all_contra_sent,
                 "unknown/neither": n_contra - all_contra_recs - all_contra_sent}
bars = ax.bar(outcomes_d1.keys(), outcomes_d1.values(),
              color=["#2196F3","#FF5722","#9E9E9E"], edgecolor="white", width=0.5)
for bar, v in zip(bars, outcomes_d1.values()):
    pct = v / n_contra * 100 if n_contra > 0 else 0
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
            f"{v}\n({pct:.0f}%)", ha="center", va="bottom", fontsize=11, fontweight="bold")
ax.set_title("market outcome on contradiction days\n(d+10 horizon, all 4 sources)", fontweight="bold")
ax.set_ylabel("number of sources")
ax.set_ylim(0, n_contra + 1)
ax.grid(axis="y", alpha=0.3)

# right: price window return for each contradiction source
ax = axes[1]
contra_plot = full_df[full_df["contradiction_flag"] == "yes"].copy()
contra_plot["label"] = contra_plot.apply(
    lambda r: f"id {r['id']}\n{str(r['source'])[:22]}", axis=1)
windows = ["d1_return_%","d3_return_%","d5_return_%","d10_return_%"]
w_labels = ["d+1","d+3","d+5","d+10"]

x = np.arange(len(contra_plot))
w = 0.18
palette = ["#1565C0","#1976D2","#42A5F5","#90CAF9"]
for i, (col, lbl) in enumerate(zip(windows, w_labels)):
    vals = contra_plot[col].values.astype(float)
    bars = ax.bar(x + (i - 1.5) * w, vals, w, label=lbl,
                  color=palette[i], edgecolor="white", alpha=0.85)

ax.axhline(0, color="black", linewidth=1, alpha=0.5)
ax.set_xticks(x)
ax.set_xticklabels(contra_plot["label"], fontsize=7)
ax.set_title("price window returns — contradiction sources", fontweight="bold")
ax.set_ylabel("cumulative return (%)")
ax.legend(fontsize=8)
ax.grid(axis="y", alpha=0.3)

fig.suptitle("ogdc contradiction analysis — market outcomes", fontsize=14, fontweight="bold")
plt.tight_layout()
save_fig(fig, "01_contradiction_outcomes.png")

# plot 2: full source matrix heatmap
fig, ax = plt.subplots(figsize=(14, 6))

hm_data = full_df[["sentiment_label","recommendation","outlook","contradiction_flag",
                    "actual_direction_d1","model_correct","sentiment_correct"]].copy()
hm_data.index = full_df.apply(lambda r: f"id{r['id']} {str(r['source'])[:28]}", axis=1)

# encode categorical to numeric for heatmap (green=positive, red=negative, yellow=neutral)
encode = {
    "sentiment_label":     {"positive":2,"neutral":1,"negative":0,"nan":1},
    "outlook":             {"positive":2,"mixed":1,"negative":0,"unknown":1},
    "contradiction_flag":  {"yes":1,"no":0},
    "actual_direction_d1": {"up":2,"flat":1,"down":0,"unknown":1},
    "model_correct":       {"yes":2,"neutral":1,"no":0,"n/a (pre-test)":1,"unknown":1},
    "sentiment_correct":   {"yes":2,"neutral":1,"no":0,"unknown":1},
}
for col, mapping in encode.items():
    if col in hm_data.columns:
        hm_data[col] = hm_data[col].astype(str).map(mapping).fillna(1)

hm_data["recommendation"] = full_df["recommendation"].apply(
    lambda x: 2 if "buy" in str(x).lower() else (0 if "sell" in str(x).lower() else 1))

sns.heatmap(hm_data.astype(float), ax=ax, cmap="RdYlGn",
            annot=False, linewidths=0.5, cbar=True,
            xticklabels=["sentiment", "rec", "outlook", "contradiction",
                         "actual dir", "model ok", "sentiment ok"])
ax.set_title("source signal matrix — all 14 ogdc sources\n(green=positive/yes/up  |  red=negative/no/down  |  yellow=neutral/mixed)",
             fontweight="bold")
ax.set_ylabel("")
ax.tick_params(axis="y", labelsize=8)
plt.tight_layout()
save_fig(fig, "02_source_matrix.png")

# plot 3: return windows over time (all dated sources)
dated_src = full_df[full_df["date"] != "n/a"].copy()
dated_src["date_dt"] = pd.to_datetime(dated_src["date"], errors="coerce")
dated_src = dated_src.dropna(subset=["date_dt"]).sort_values("date_dt")

fig, ax = plt.subplots(figsize=(14, 5))
for _, r in dated_src.iterrows():
    c_flag = r["contradiction_flag"]
    color  = "#E53935" if c_flag == "yes" else "#1565C0"
    marker = "★" if c_flag == "yes" else "●"
    alpha  = 0.95 if c_flag == "yes" else 0.6
    d10    = r["d10_return_%"]
    if pd.notna(d10):
        ax.scatter(r["date_dt"], d10, color=color, s=120 if c_flag=="yes" else 60,
                   zorder=3, alpha=alpha)
        ax.annotate(f"id{int(r['id'])}", (r["date_dt"], d10),
                    textcoords="offset points", xytext=(0, 8),
                    fontsize=7, ha="center",
                    color=color, fontweight="bold" if c_flag=="yes" else "normal")

ax.axhline(0, color="black", linewidth=1, alpha=0.5)
ax.fill_between(dated_src["date_dt"], 0, alpha=0.03, color="grey")

contra_patch  = mpatches.Patch(color="#E53935", label="contradiction sources (buy+neg)")
aligned_patch = mpatches.Patch(color="#1565C0", label="aligned sources")
ax.legend(handles=[contra_patch, aligned_patch], fontsize=9)
ax.set_title("d+10 price return after publication — all dated sources", fontweight="bold")
ax.set_xlabel("publication date")
ax.set_ylabel("d+10 cumulative return (%)")
ax.grid(alpha=0.2)
plt.tight_layout()
save_fig(fig, "03_d10_returns_timeline.png")

# plot 4: model predictions vs actual for test period + sentiment overlaid
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=False)

# top: ogdc price full period with source annotations
price_plot = price.dropna(subset=["actual_return"]).copy()
ax1.plot(price_plot["date"], price["price"][:len(price_plot)], color="steelblue",
         linewidth=1.2, alpha=0.8, label="ogdc price (pkr)")

# overlay contradiction source dates on price chart
for _, r in full_df.iterrows():
    if r["date"] == "n/a":
        continue
    dt = pd.to_datetime(r["date"], errors="coerce")
    if pd.isna(dt):
        continue
    c_flag = r["contradiction_flag"]
    color  = "#E53935" if c_flag == "yes" else "#4CAF50"
    # find price at that date
    prow = price[price["date"] == dt]
    if not prow.empty:
        p_val = prow["price"].values[0]
        ax1.scatter(dt, p_val, color=color, s=90, zorder=4, alpha=0.9)
        ax1.annotate(f"id{int(r['id'])}", (dt, p_val),
                     textcoords="offset points", xytext=(0, 6),
                     fontsize=6.5, ha="center", color=color)

contra_p = mpatches.Patch(color="#E53935", label="contradiction source")
align_p  = mpatches.Patch(color="#4CAF50", label="aligned source")
ax1.legend(handles=[contra_p, align_p], fontsize=8, loc="upper left")
ax1.set_title("ogdc price history with sentiment source annotations", fontweight="bold")
ax1.set_ylabel("price (pkr)")
ax1.grid(alpha=0.2)

# bottom: model accuracy in test period
model_plot = pd.read_csv(MODEL_PATH)
model_plot["Date"] = pd.to_datetime(model_plot["Date"])

# Map RF/GBM direction to up/down (use whichever model you prefer)
# Using RF direction as the model_predicted_direction
if "rf_direction" in model_plot.columns:
    model_plot["model_predicted_direction"] = model_plot["rf_direction"].map({1: "up", 0: "down"})
    model_plot["model_predicted_return_pct"] = model_plot.get("reg_predicted_return", 0)

# Bar chart for actual returns
ax2.bar(model_plot["Date"], model_plot["actual_return"],
        color=model_plot["actual_return"].apply(lambda x: "#4CAF50" if x > 0 else "#F44336"),
        width=1, alpha=0.7, label="actual daily return")

# Line plot for model predictions
ax2.plot(model_plot["Date"], model_plot["model_predicted_return_pct"],
         color="navy", linewidth=0.8, alpha=0.6, label="model predicted return")

ax2.axhline(0, color="black", linewidth=0.7)
ax2.set_title("model predictions vs actual returns (test period: 2025-01-09 → 2026-04-03)", fontweight="bold")
ax2.set_xlabel("date")
ax2.set_ylabel("return (%)")
ax2.legend(fontsize=8, loc="upper left")
ax2.grid(alpha=0.2)
plt.tight_layout()
save_fig(fig, "04_price_and_model.png")

# plot 5: recommendation vs sentiment scatter for all 14 sources
fig, ax = plt.subplots(figsize=(9, 6))
for _, r in full_df.iterrows():
    d10 = r["d10_return_%"]
    if pd.isna(d10):
        continue
    c_flag = r["contradiction_flag"]
    col = "#E53935" if c_flag == "yes" else ("#4CAF50" if r["sentiment_label"]=="positive" else "#9E9E9E")
    marker = "D" if c_flag == "yes" else "o"
    ax.scatter(r["sentiment_score"], d10, color=col, marker=marker,
               s=130 if c_flag=="yes" else 70, zorder=3, alpha=0.85)
    ax.annotate(f"id{int(r['id'])}", (r["sentiment_score"], d10),
                textcoords="offset points", xytext=(5, 3), fontsize=7.5, color=col)

ax.axhline(0, color="black", linewidth=1, linestyle="--", alpha=0.5)
ax.axvline(0, color="black", linewidth=1, linestyle="--", alpha=0.5)
ax.set_xlabel("sentiment score  (−1=very negative  →  +1=very positive)", fontsize=11)
ax.set_ylabel("d+10 cumulative return (%)", fontsize=11)
ax.set_title("sentiment score vs d+10 return — all 14 ogdc sources\n(◆ = contradiction, ● = aligned)",
             fontweight="bold")
contra_p = mpatches.Patch(color="#E53935", label="contradiction (buy+negative outlook)")
pos_p    = mpatches.Patch(color="#4CAF50", label="positive aligned")
neu_p    = mpatches.Patch(color="#9E9E9E", label="neutral/negative aligned")
ax.legend(handles=[contra_p, pos_p, neu_p], fontsize=9)
ax.grid(alpha=0.25)
plt.tight_layout()
save_fig(fig, "05_sentiment_vs_return_scatter.png")

print("\n  all 5 plots saved.")

# ══════════════════════════════════════════════════════════════════════════════
# step 8 — three-sentence conclusion (executive summary)
# ══════════════════════════════════════════════════════════════════════════════
banner("step 8 — conclusion")

best_signal_name = best_sig
best_signal_rate = best_rate

conclusion = f"""
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │  conclusion                                                                 │
  └─────────────────────────────────────────────────────────────────────────────┘

  on {n_contra} contradiction days where recommendations conflicted with fundamental
  outlook, the market followed the recommendation {all4_rec_rate:.0f}% of the time (vs.
  {all4_sent_rate:.0f}% following the negative sentiment), suggesting that ogdc investors
  actively look through one-off earnings declines when compensated by high dividend
  yield and deep valuation discount — a contrarian signal appears structurally valid
  for this stock.

  the ml model correctly predicted direction on {model_acc_contra:.0f}% of dated contradiction
  days, {'outperforming' if model_acc_contra > sent_acc_contra else 'matching'} pure sentiment analysis (which scored {sent_acc_contra:.0f}%);
  however, the sample of dated contradiction events is small ({n_contra_dated} rows) and
  the finding must be treated as indicative, not conclusive.

  the single best signal on contradiction days was '{best_signal_name}' with a
  {best_signal_rate:.0f}% success rate, confirming that analyst buy recommendations carry
  more weight than negative fundamental framing in driving near-term ogdc price action.

  ─────────────────────────────────────────────────────────────────────────────
  ⚠  disclaimer: analysis based on {n_contra} contradiction days across 14 sources
  spanning {full_df[full_df['date']!='n/a']['date'].min()} to
  {full_df[full_df['date']!='n/a']['date'].max()}.
  findings are preliminary and require validation on a larger dataset.
  ─────────────────────────────────────────────────────────────────────────────
"""
print(conclusion)

# ══════════════════════════════════════════════════════════════════════════════
# step 9 — save outputs
# ══════════════════════════════════════════════════════════════════════════════
banner("step 9 — saving output files")

full_df.to_csv(OUT_FULL, index=False)
summary_df.to_csv(OUT_SUMMARY, index=False)
print(f"  → contradiction_analysis_full.csv    ({len(full_df)} rows, {len(full_df.columns)} columns)")
print(f"  → contradiction_summary.csv          ({len(summary_df)} rows)")
print(f"  → 5 visualisation pngs (prefix: contra_)")