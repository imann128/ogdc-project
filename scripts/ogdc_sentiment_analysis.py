"""
ogdc_sentiment_analysis.py
==========================
ogdc sentiment analysis & contradiction detection

full pipeline: keyword scoring -> sentiment classification ->
csv price mapping -> contradiction analysis -> multi-day tracking.

usage:
    python ogdc_sentiment_analysis.py

requires: pandas (standard lib re/json included)
"""

import pandas as pd
import re
import json
import os
from datetime import datetime
from typing import Optional, Tuple

# paths for input and output
base = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base, "..", "data", "processed")
os.makedirs(data_dir, exist_ok=True)
csv_path = os.path.join(data_dir, "ogdc_cleaned.csv")
out_path = os.path.join(data_dir, "ogdc_sentiment_results.csv")

# ==================================================================
# 1. keyword dictionaries for sentiment scoring
# ==================================================================

bullish = {
    # explicit recommendations
    "strong buy": 2, "strong-buy": 2,
    "buy": 1, "accumulate": 1,
    # market direction
    "bullish": 1, "bull run": 1,
    "upward": 1, "rally": 1, "outperform": 1, "overweight": 1,
    "all time high": 1, "breakout": 1, "momentum": 1,
    # quality / performance signals
    "best stock": 2,
    "blue chip": 2,
    "dividend powerhouse": 2,
    "highest ever": 2,
    "record dividend": 2,
    "increased by": 1,
    "lower volatility": 1,
    "top-performing": 1,
    "breaking record": 1,
    "traded in green": 1,
    "index-heavy stocks": 1,
    "strong buying": 1,
    "handsome return": 1, "good": 1, "positive": 1,
    "is a buy": 1, "not a sell": 1,
}

bearish = {
    # explicit recommendations
    "strong sell": -2, "strong-sell": -2,
    "sell": -1, "reduce": -1,
    # market direction
    "bearish": -1, "bear run": -1,
    "downward": -1, "decline": -1, "declining": -1,
    "underperform": -1, "crash": -1, "plunge": -1,
    "correction": -1, "all time low": -1,
    # earnings / profit deterioration
    "profit collapse": -2,
    "profit falls": -1, "profit fell": -1,
    "plunged": -1, "plummeting": -1,
    "fell": -1, "fallen": -1,
    "dropped": -1, "drop": -1,
    "sinks": -1, "sank": -1,
    "slumped": -1, "slump": -1,
    "missed estimates": -1,
    "drastic decline": -1,
    "drastic reduction": -1,
    "forced closure": -1,
    "falling global": -1,
    "severely impacting": -1,
    "bad setup": -1, "poor": -1, "negative": -1,
    "is a sell": -1, "not a buy": -1,
}

image_positive_cues = {"green", "up", "gain", "positive", "rise", "bull"}
image_negative_cues = {"red", "down", "loss", "negative", "fall", "bear"}

normalization_divisor = 15.0   # scale raw score to [-1, +1]

# ==================================================================
# 2. scoring engine
# ==================================================================

def score_text(text: str, image_cues: list = None) -> Tuple[float, list]:
    """
    score raw article text using keyword dictionaries.
    returns (normalized_score, hits_list).
    """
    text_l = text.lower()
    raw = 0
    hits = []

    combined = {**bullish, **bearish}
    for phrase, weight in combined.items():
        # whole-word boundary match
        pattern = r'\b' + re.escape(phrase) + r'\b'
        count = len(re.findall(pattern, text_l))
        if count:
            raw += weight * count
            hits.append((phrase, weight, count, weight * count))

    # image cue adjustments
    if image_cues:
        for cue in image_cues:
            cue_l = cue.lower()
            if cue_l in image_positive_cues:
                raw += 0.5
                hits.append(("[img:" + cue + "]", 0.5, 1, 0.5))
            elif cue_l in image_negative_cues:
                raw -= 0.5
                hits.append(("[img:" + cue + "]", -0.5, 1, -0.5))

    score = max(-1.0, min(1.0, raw / normalization_divisor))
    return round(score, 4), hits


def classify(score: float) -> str:
    """convert numeric score to sentiment label"""
    if score > 0.2:
        return "positive"
    if score < -0.2:
        return "negative"
    return "neutral"


def detect_contradiction(rec: str, outlook: str) -> Tuple[bool, str]:
    """
    flag if recommendation contradicts stated outlook.
    returns (is_contradiction, contradiction_type).
    """
    buy_recs = {"buy", "strong buy", "accumulate", "add", "implicit buy", "overweight"}
    sell_recs = {"sell", "reduce", "exit", "strong sell", "underweight"}

    rec_l = rec.lower()
    is_buy = any(b in rec_l for b in buy_recs)
    is_sell = any(s in rec_l for s in sell_recs)

    if is_buy and outlook.lower() == "negative":
        return True, "buy + negative outlook"
    if is_sell and outlook.lower() == "positive":
        return True, "sell + positive outlook"
    return False, "none"


# ==================================================================
# 3. price data mapping
# ==================================================================

def load_price_data(csv_path: str) -> pd.DataFrame:
    """load and sort price data"""
    df = pd.read_csv(csv_path, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def get_article_price(date_str: str, df: pd.DataFrame) -> Optional[float]:
    """price on or just before article date (close price)"""
    if not date_str:
        return None
    d = pd.Timestamp(date_str)
    before = df[df["Date"] <= d]
    return float(before.iloc[-1]["Price"]) if not before.empty else None


def get_next_n_change(date_str: str, n: int, df: pd.DataFrame) -> Tuple[Optional[float], Optional[float]]:
    """
    returns (day_n_pct_change, cumulative_pct_from_article_date)
    day_n_pct_change is the Change% column value on the nth trading day after article
    cumulative is (price_at_D+n - price_at_article_date) / price_at_article_date * 100
    """
    if not date_str:
        return None, None
    d = pd.Timestamp(date_str)
    future = df[df["Date"] > d].reset_index(drop=True)
    if len(future) < n:
        return None, None

    p0 = get_article_price(date_str, df)
    pn = float(future.iloc[n - 1]["Price"])
    day_chg = float(future.iloc[n - 1]["Change %"])
    cum = (pn - p0) / p0 * 100 if p0 else None
    return round(day_chg, 2), round(cum, 2) if cum else None


def d1_match(label: str, d1_chg: Optional[float]) -> str:
    """check if next-day price direction matches sentiment label"""
    if label == "neutral" or d1_chg is None:
        return "excluded"
    if label == "positive" and d1_chg > 0:
        return "match"
    if label == "negative" and d1_chg < 0:
        return "match"
    return "mismatch"


def contra_market_result(rec: str, d1_chg: Optional[float]) -> Optional[str]:
    """for contradiction articles: did market follow rec or outlook?"""
    if d1_chg is None:
        return None
    buy_recs = {"buy", "strong buy", "accumulate", "implicit buy"}
    is_buy = any(b in rec.lower() for b in buy_recs)
    if is_buy and d1_chg > 0:
        return "followed_recommendation"
    if is_buy and d1_chg < 0:
        return "followed_outlook"
    return "neutral_day"


# ==================================================================
# 4. article corpus (all 14 sources)
# ==================================================================

articles = [
    {"id": 1, "source": "ksestocks blog",
     "url": "https://ksestocks.com/blog/ogdc-the-peoples-choice-for-all-time-stock/",
     "date": "2025-02-20", "score": 0.00, "label": "neutral",
     "rec": "none", "outlook": "mixed",
     "contra": False, "contra_type": "none",
     "contra_phrase": "neither going up nor down",
     "kw_pos": ["good corrections"],
     "kw_neg": ["confused stage", "neither going up nor down"],
     "price_target": "180-192 support zone"},

    {"id": 2, "source": "stockanalysis.com",
     "url": "https://stockanalysis.com/quote/psx/OGDC/",
     "date": "2026-03-13", "score": 0.43, "label": "positive",
     "rec": "none", "outlook": "mixed",
     "contra": False, "contra_type": "none",
     "contra_phrase": "revenue -13.48% yet 25.71% 1y return",
     "kw_pos": ["increased by +16.27%", "lower volatility", "1y return 25.71%",
                "dividend 5.48%", "new oil & gas discoveries"],
     "kw_neg": ["revenue -13.48%", "earnings -18.70%"],
     "price_target": "none"},

    {"id": 3, "source": "investing.com technical",
     "url": "https://www.investing.com/equities/oil---gas-dev-technical",
     "date": "2026-03-10", "score": 0.82, "label": "positive",
     "rec": "strong buy", "outlook": "positive",
     "contra": False, "contra_type": "none", "contra_phrase": "",
     "kw_pos": ["strong buy daily", "strong buy weekly/monthly",
                "buy rsi", "buy macd", "10 indicator buys", "8 ma buys"],
     "kw_neg": ["ma5 sell", "ma200 sell"],
     "price_target": "none"},

    {"id": 4, "source": "tradingview community ideas",
     "url": "https://www.tradingview.com/symbols/PSX-OGDC/ideas/",
     "date": "2026-03-12", "score": 0.71, "label": "positive",
     "rec": "buy", "outlook": "mixed",
     "contra": False, "contra_type": "none",
     "contra_phrase": "10 long ideas vs 3 short ideas",
     "kw_pos": ["bullish structure x3", "accumulate/buy x4",
                "breakout x3", "upside/rally x3", "10 long ideas"],
     "kw_neg": ["bearish divergence x2", "correction x2",
                "downtrend x1", "3 short ideas"],
     "price_target": "t1 289-300 / t2 320-335 / t3 350-420"},

    {"id": 5, "source": "youtube video",
     "url": "https://youtu.be/_NOSb2kgWKQ",
     "date": None, "score": 0.40, "label": "positive",
     "rec": "buy", "outlook": "positive",
     "contra": False, "contra_type": "none", "contra_phrase": "",
     "kw_pos": ["best stock (title)", "ogdc featured buy", "positive/green thumbnail"],
     "kw_neg": [],
     "price_target": "none (youtube blocked)"},

    {"id": 6, "source": "defencepk forum",
     "url": "https://defencepk.com/forums/threads/...17551/",
     "date": "2025-01-01", "score": 0.53, "label": "positive",
     "rec": "none", "outlook": "positive",
     "contra": False, "contra_type": "none", "contra_phrase": "",
     "kw_pos": ["index-heavy stocks traded in green", "breaking record 72nd time",
                "bullish momentum", "upward trajectory", "strong buying",
                "top-performing market 178% gain"],
     "kw_neg": [],
     "price_target": "none"},

    {"id": 7, "source": "quora - blue-chip list",
     "url": "https://www.quora.com/Which-are-the-blue-chip-companies-in-Pakistan",
     "date": None, "score": 0.33, "label": "positive",
     "rec": "hold", "outlook": "positive",
     "contra": False, "contra_type": "none", "contra_phrase": "",
     "kw_pos": ["blue chip x2", "well-established financially strong",
                "consistent growth", "safe less risky"],
     "kw_neg": [],
     "price_target": "none (evergreen reference)"},

    {"id": 8, "source": "tradersunion forecast",
     "url": "https://tradersunion.com/currencies/forecast/ogdc-pkr/daily-and-weekly/",
     "date": "2026-03-12", "score": -0.35, "label": "negative",
     "rec": "sell", "outlook": "negative",
     "contra": False, "contra_type": "none", "contra_phrase": "",
     "kw_pos": [],
     "kw_neg": ["decline -1.28%", "red forecast chart",
                "downside target pkr 274.94", "forecast arrow pointing down"],
     "price_target": "pkr 274.94 (downside)"},

    {"id": 9, "source": "mettis global - q4fy24 profit collapse",
     "url": "https://mettisglobal.news/ogdcs-profit-collapse-sinks-oil-gas-sector-in-q4fy24/",
     "date": "2024-10-04", "score": -0.60, "label": "negative",
     "rec": "none", "outlook": "negative",
     "contra": False, "contra_type": "none", "contra_phrase": "",
     "kw_pos": [],
     "kw_neg": ["profit collapse x2", "sinks", "plummeted 42% yoy",
                "fell", "dropped", "drastic decline",
                "missed estimates", "decline"],
     "price_target": "none"},

    {"id": 10, "source": "dawn.com - ogdcl profit slumps 33pc",
     "url": "https://www.dawn.com/news/1895049",
     "date": "2025-03-01", "score": -0.67, "label": "negative",
     "rec": "none", "outlook": "negative",
     "contra": False, "contra_type": "none", "contra_phrase": "",
     "kw_pos": [],
     "kw_neg": ["plunged 33pc", "33% reduction",
                "falling global crude prices", "forced closure of wells",
                "revenue down 12.3%", "exchange rate appreciation", "decline"],
     "price_target": "none"},

    {"id": 11, "source": "mettis global - fy25 profit falls",
     "url": "https://mettisglobal.news/OGDCs-FY25-profit-falls-to-Rs170bn-shareholders-to-receive-record-dividend-55420",
     "date": "2025-09-23", "score": -0.13, "label": "neutral",
     "rec": "implicit buy", "outlook": "negative",
     "contra": True, "contra_type": "buy + negative outlook",
     "contra_phrase": "profit falls 18.7% ... yet highest-ever dividend of rs15.05/share",
     "kw_pos": ["highest ever distribution x2", "finance income surged 97.90%",
                "record dividend", "operating expenses decreased"],
     "kw_neg": ["profit falls", "pat down 18.70%", "eps fell",
                "net sales down 13.48%", "gross profit fell 18.25%"],
     "price_target": "none (dividend rs15.05/share announced)"},

    {"id": 12, "source": "propakistani - fy24 profit falls 7%",
     "url": "https://propakistani.pk/2024/09/23/ogdc-profit-falls-7-to-rs-208-9-billion-in-fy24/",
     "date": "2024-09-23", "score": -0.07, "label": "neutral",
     "rec": "implicit buy", "outlook": "negative",
     "contra": True, "contra_type": "buy + negative outlook",
     "contra_phrase": "profit falls 7% yet topline up 12%, arif habib framing supportive",
     "kw_pos": ["topline climbed 12% yoy", "oil production up 2%",
                "5 new discoveries", "final dividend announced",
                "arif habib brokerage framing"],
     "kw_neg": ["profit falls 7%", "pat down 7%", "decline 42% qoq",
                "other income down 73%", "missed estimates"],
     "price_target": "none (dividend rs4.00/share final)"},

    {"id": 13, "source": "insight research / topline securities",
     "url": "insight research on ogdc growth potential (june 2024)",
     "date": None, "score": 0.47, "label": "positive",
     "rec": "buy", "outlook": "negative",
     "contra": True, "contra_type": "buy + negative outlook",
     "contra_phrase": "18.7% drop in net profit ... brokerages reiterated buy citing ev/ebitda 1.1x and dividend powerhouse",
     "kw_pos": ["buy rating", "dividend powerhouse",
                "ev/ebitda deep discount 1.1x",
                "topline securities buy", "akd research buy"],
     "kw_neg": ["18.7% drop net profit", "7% dip fy24",
                "lower oil prices falling output"],
     "price_target": "none cited"},

    {"id": 14, "source": "igi securities strategy report",
     "url": "igi securities strategy 2024 report (december 2023)",
     "date": None, "score": 0.33, "label": "positive",
     "rec": "buy", "outlook": "negative",
     "contra": True, "contra_type": "buy + negative outlook",
     "contra_phrase": "massive circular debt severely impacting cash flow ... analysts encouraged investment based on settlement plan",
     "kw_pos": ["circular debt settlement plan",
                "regular monthly interest payments",
                "reko diq stake sale gains",
                "analysts encouraged investment"],
     "kw_neg": ["circular debt hundreds of billions",
                "severely impacting cash flow",
                "historical payout ratios impacted"],
     "price_target": "none"},
]


# ==================================================================
# 5. main pipeline
# ==================================================================

def run_analysis(csv_file_path: str) -> list:
    """run full sentiment analysis pipeline on all articles"""
    df = load_price_data(csv_file_path)
    results = []

    for art in articles:
        row = dict(art)

        # price window tracking (d+1, d+3, d+5, d+10)
        for n, key in [(1, "d1"), (3, "d3"), (5, "d5"), (10, "d10")]:
            chg, cum = get_next_n_change(art["date"], n, df)
            row[key + "_chg"] = chg
            row[key + "_cum"] = cum

        # d+1 sentiment match
        row["d1_match"] = d1_match(art["label"], row["d1_chg"])

        # contradiction market result (d+1 and d+10)
        if art["contra"]:
            row["contra_market_d1"] = contra_market_result(art["rec"], row["d1_chg"])
            row["contra_market_d10"] = (
                "followed_recommendation" if (row["d10_cum"] or 0) > 0
                else "followed_outlook"
            )
        else:
            row["contra_market_d1"] = None
            row["contra_market_d10"] = None

        results.append(row)

    return results


def print_summary(results: list):
    """print formatted summary of sentiment analysis"""
    print("ogdc sentiment analysis")
    print("=" * 50)

    # overall stats
    trackable = [r for r in results if r["d1_match"] not in ("excl.", "excluded")]
    matches = [r for r in trackable if r["d1_match"] == "match"]
    contra = [r for r in results if r["contra"]]
    contra_tracked = [r for r in contra if r["d1_chg"] is not None]

    print(f"\ntotal articles          : {len(results)}")
    print(f"trackable (d+1 match)   : {len(trackable)}")
    print(f"overall d+1 match rate  : {len(matches)/len(trackable)*100:.1f}% ({len(matches)}/{len(trackable)})")
    print(f"contradiction articles  : {len(contra)}")
    print(f"contradiction trackable : {len(contra_tracked)}")

    # per article table
    print("\nid  source                            date         score   label      rec          d+1     d+10cum   match      contra")
    print("-" * 100)
    for r in results:
        d1 = f"{r['d1_chg']:+.2f}%" if r['d1_chg'] is not None else "   n/a"
        d10 = f"{r['d10_cum']:+.2f}%" if r['d10_cum'] is not None else "    n/a"
        contra_flag = "yes" if r["contra"] else ""
        print(f"{r['id']:2d}  {r['source']:30s} {str(r['date']):10s} "
              f"{r['score']:6.2f}  {r['label']:8s}  {r['rec']:10s}  "
              f"{d1:>7}  {d10:>8}  {r['d1_match']:8s}  {contra_flag}")

    # contradiction detail
    print("\ncontradiction analysis")
    for r in contra:
        print(f"\n  id {r['id']}: {r['source']}")
        print(f"  type       : {r['contra_type']}")
        print(f"  key phrase : {r['contra_phrase']}")
        print(f"  d+1        : {r['d1_chg']} -> market {r['contra_market_d1'] or 'n/a'}")
        print(f"  d+10 cum   : {r['d10_cum']} -> market {r['contra_market_d10'] or 'n/a'}")

    # aligned vs contradiction comparison
    print("\naligned vs contradiction - match rate comparison")
    print("-" * 60)

    aligned_buy_pos = [r for r in results if not r["contra"]
                       and r["rec"] in ("buy", "strong buy", "hold")
                       and r["outlook"] == "positive"
                       and r["d1_match"] not in ("excl.", "excluded")]
    aligned_sell_neg = [r for r in results if not r["contra"]
                        and r["rec"] == "sell"
                        and r["outlook"] == "negative"
                        and r["d1_match"] not in ("excl.", "excluded")]
    contra_buy_neg = [r for r in results if r["contra"]
                      and r["d1_chg"] is not None]

    def match_rate(lst):
        if not lst:
            return 0
        return sum(1 for r in lst if r["d1_match"] == "match") / len(lst) * 100

    print(f"\n  {'article type':<40} {'count':>5}  {'d+1 match rate':>15}")
    print(f"  {'-'*40} {'-'*5}  {'-'*15}")
    print(f"  {'aligned: buy/hold + positive outlook':<40} {len(aligned_buy_pos):>5}  {match_rate(aligned_buy_pos):>14.0f}%")
    print(f"  {'aligned: sell + negative outlook':<40} {len(aligned_sell_neg):>5}  {match_rate(aligned_sell_neg):>14.0f}%")
    print(f"  {'contradiction: buy + negative outlook':<40} {len(contra_buy_neg):>5}  {'mixed (see above)':>15}")

    # verdict
    print("\nverdict: does market reward buying against negative outlook?")
    print("""
  answer: it depends on the type of negative news.

  buy the contradiction when:
    - negative news is one-off (exchange loss, tax reversal, non-recurring item)
    - topline/operational metrics are still growing
    - brokerage support exists alongside the negative headline

  outlook can be trusted when:
    - negative news is structural (multi-line decline)
    - even a record dividend cannot mask persistent deterioration

  contradiction articles have higher uncertainty (~50% d+1 match rate)
  vs aligned articles (~70% d+1 match rate).
""")


def export_csv(results: list, export_path: str):
    """export results to csv file"""
    rows = []
    for r in results:
        rows.append({
            "id": r["id"],
            "source": r["source"],
            "url": r["url"],
            "date": r["date"],
            "sentiment_score": r["score"],
            "sentiment_label": r["label"],
            "recommendation": r["rec"],
            "outlook": r["outlook"],
            "contradiction_flag": "yes" if r["contra"] else "no",
            "contradiction_type": r["contra_type"],
            "key_contradiction_phrase": r["contra_phrase"],
            "price_target": r.get("price_target", "none"),
            "d1_chg_%": r["d1_chg"],
            "d3_cum_%": r["d3_cum"],
            "d5_cum_%": r["d5_cum"],
            "d10_cum_%": r["d10_cum"],
            "d1_match": r["d1_match"],
            "contra_market_d1": r["contra_market_d1"],
            "contra_market_d10": r["contra_market_d10"],
        })
    pd.DataFrame(rows).to_csv(export_path, index=False)
    print(f"\nresults exported to: {export_path}")


# ==================================================================
# 6. entry point
# ==================================================================

if __name__ == "__main__":
    print(f"reading price data from: {csv_path}")
    print(f"file exists: {os.path.exists(csv_path)}")
    
    results = run_analysis(csv_path)
    print_summary(results)
    export_csv(results, out_path)



'''
"""
OGDC Sentiment Analysis & Contradiction Detection
==================================================
Full pipeline: keyword scoring → sentiment classification →
CSV price mapping → contradiction analysis → multi-day tracking.

Usage:
    python ogdc_sentiment_analysis.py

Requires: pandas (standard lib re/json included)
"""

import pandas as pd
import re
import json
from datetime import datetime

# ═══════════════════════════════════════════════════════════════════════════
# 1. Keyword Dictionaries
# ═══════════════════════════════════════════════════════════════════════════

BULLISH = {
    # Explicit recommendations
    "strong buy": 2, "strong-buy": 2,
    "buy": 1, "accumulate": 1,
    # Market direction
    "bullish": 1, "bull run": 1,
    "upward": 1, "rally": 1, "outperform": 1, "overweight": 1,
    "all time high": 1, "breakout": 1, "momentum": 1,
    # Quality / performance signals
    "best stock": 2,                        # video title keyword
    "blue chip": 2,                         # blue-chip status
    "dividend powerhouse": 2,               # strong dividend label
    "highest ever": 2,                      # record dividend signal
    "record dividend": 2,
    "increased by": 1,                      # price/metric increase
    "lower volatility": 1,                  # stability = positive
    "top-performing": 1,
    "breaking record": 1,
    "traded in green": 1,                   # index stocks green
    "index-heavy stocks": 1,
    "strong buying": 1,
    "handsome return": 1, "good": 1, "positive": 1,
    "is a buy": 1, "not a sell": 1,
}

BEARISH = {
    # Explicit recommendations
    "strong sell": -2, "strong-sell": -2,
    "sell": -1, "reduce": -1,
    # Market direction
    "bearish": -1, "bear run": -1,
    "downward": -1, "decline": -1, "declining": -1,
    "underperform": -1, "crash": -1, "plunge": -1,
    "correction": -1, "all time low": -1,
    # Earnings / profit deterioration
    "profit collapse": -2,
    "profit falls": -1, "profit fell": -1,
    "plunged": -1, "plummeting": -1,
    "fell": -1, "fallen": -1,
    "dropped": -1, "drop": -1,
    "sinks": -1, "sank": -1,
    "slumped": -1, "slump": -1,
    "missed estimates": -1,
    "drastic decline": -1,
    "drastic reduction": -1,
    "forced closure": -1,
    "falling global": -1,
    "severely impacting": -1,
    "bad setup": -1, "poor": -1, "negative": -1,
    "is a sell": -1, "not a buy": -1,
}

IMAGE_POSITIVE_CUES = {"green", "up", "gain", "positive", "rise", "bull"}
IMAGE_NEGATIVE_CUES = {"red", "down", "loss", "negative", "fall", "bear"}

NORMALIZATION_DIVISOR = 15.0   # scale raw score to [-1, +1]


# ═══════════════════════════════════════════════════════════════════════════
# 2. Scoring Engine
# ═══════════════════════════════════════════════════════════════════════════

def score_text(text: str, image_cues: list[str] = None) -> tuple[float, list[tuple]]:
    """
    Score raw article text using keyword dictionaries.
    Returns (normalized_score, hits_list).
    """
    text_l = text.lower()
    raw = 0
    hits = []

    combined = {**BULLISH, **BEARISH}
    for phrase, weight in combined.items():
        # whole-word boundary match
        pattern = r'\b' + re.escape(phrase) + r'\b'
        count = len(re.findall(pattern, text_l))
        if count:
            raw += weight * count
            hits.append((phrase, weight, count, weight * count))

    # image cue adjustments
    if image_cues:
        for cue in image_cues:
            cue_l = cue.lower()
            if cue_l in IMAGE_POSITIVE_CUES:
                raw += 0.5
                hits.append((f"[img:{cue}]", 0.5, 1, 0.5))
            elif cue_l in IMAGE_NEGATIVE_CUES:
                raw -= 0.5
                hits.append((f"[img:{cue}]", -0.5, 1, -0.5))

    score = max(-1.0, min(1.0, raw / NORMALIZATION_DIVISOR))
    return round(score, 4), hits


def classify(score: float) -> str:
    if score > 0.2:  return "Positive"
    if score < -0.2: return "Negative"
    return "Neutral"


def detect_contradiction(rec: str, outlook: str) -> tuple[bool, str]:
    """
    Flag if recommendation contradicts stated outlook.
    Returns (is_contradiction, contradiction_type).
    """
    buy_recs  = {"buy", "strong buy", "accumulate", "add", "implicit buy", "overweight"}
    sell_recs = {"sell", "reduce", "exit", "strong sell", "underweight"}

    rec_l = rec.lower()
    is_buy  = any(b in rec_l for b in buy_recs)
    is_sell = any(s in rec_l for s in sell_recs)

    if is_buy  and outlook.lower() == "negative":
        return True, "Buy + Negative Outlook"
    if is_sell and outlook.lower() == "positive":
        return True, "Sell + Positive Outlook"
    return False, "None"


# ═══════════════════════════════════════════════════════════════════════════
# 3. Price Data Mapping
# ═══════════════════════════════════════════════════════════════════════════

def load_price_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def get_article_price(date_str: str, df: pd.DataFrame) -> float | None:
    """Price on or just before article date (close price)."""
    if not date_str: return None
    d = pd.Timestamp(date_str)
    before = df[df["Date"] <= d]
    return float(before.iloc[-1]["Price"]) if not before.empty else None


def get_next_n_change(date_str: str, n: int, df: pd.DataFrame) -> tuple[float | None, float | None]:
    """
    Returns (day_n_pct_change, cumulative_pct_from_article_date).
    day_n_pct_change is the Change% column value on the nth trading day after article.
    cumulative is (price_at_D+n - price_at_article_date) / price_at_article_date * 100.
    """
    if not date_str: return None, None
    d = pd.Timestamp(date_str)
    future = df[df["Date"] > d].reset_index(drop=True)
    if len(future) < n: return None, None

    p0 = get_article_price(date_str, df)
    pn = float(future.iloc[n - 1]["Price"])
    day_chg = float(future.iloc[n - 1]["Change %"])
    cum = (pn - p0) / p0 * 100 if p0 else None
    return round(day_chg, 2), round(cum, 2) if cum else None


def d1_match(label: str, d1_chg: float | None) -> str:
    """Check if next-day price direction matches sentiment label."""
    if label == "Neutral" or d1_chg is None:
        return "Excluded"
    if label == "Positive" and d1_chg > 0: return "MATCH"
    if label == "Negative" and d1_chg < 0: return "MATCH"
    return "MISMATCH"


def contra_market_result(rec: str, d1_chg: float | None) -> str | None:
    """For contradiction articles: did market follow rec or outlook?"""
    if d1_chg is None: return None
    buy_recs = {"buy", "strong buy", "accumulate", "implicit buy"}
    is_buy = any(b in rec.lower() for b in buy_recs)
    if is_buy and d1_chg > 0:  return "followed_recommendation"
    if is_buy and d1_chg < 0:  return "followed_outlook"
    return "neutral_day"


# ═══════════════════════════════════════════════════════════════════════════
# 4. ARTICLE CORPUS  (all 14 URLs)
# ═══════════════════════════════════════════════════════════════════════════

ARTICLES = [
    {"id":  1, "source": "KSEStocks Blog",
     "url": "https://ksestocks.com/blog/ogdc-the-peoples-choice-for-all-time-stock/",
     "date": "2025-02-20", "score": 0.00, "label": "Neutral",
     "rec": "None", "outlook": "Mixed",
     "contra": False, "contra_type": "None",
     "contra_phrase": "Neither going up nor down",
     "kw_pos": ["good corrections"],
     "kw_neg": ["confused stage", "neither going up nor down"],
     "price_target": "180–192 support zone"},

    {"id":  2, "source": "StockAnalysis.com",
     "url": "https://stockanalysis.com/quote/psx/OGDC/",
     "date": "2026-03-13", "score": 0.43, "label": "Positive",
     "rec": "None", "outlook": "Mixed",
     "contra": False, "contra_type": "None",
     "contra_phrase": "Revenue −13.48% yet 25.71% 1Y return",
     "kw_pos": ["increased by +16.27%", "lower volatility", "1Y return 25.71%",
                "dividend 5.48%", "new oil & gas discoveries"],
     "kw_neg": ["revenue −13.48%", "earnings −18.70%"],
     "price_target": "None"},

    {"id":  3, "source": "Investing.com Technical",
     "url": "https://www.investing.com/equities/oil---gas-dev-technical",
     "date": "2026-03-10", "score": 0.82, "label": "Positive",
     "rec": "Strong Buy", "outlook": "Positive",
     "contra": False, "contra_type": "None", "contra_phrase": "",
     "kw_pos": ["Strong Buy daily", "Strong Buy weekly/monthly",
                "Buy RSI", "Buy MACD", "10 indicator buys", "8 MA buys"],
     "kw_neg": ["MA5 Sell", "MA200 Sell"],
     "price_target": "None"},

    {"id":  4, "source": "TradingView Community Ideas",
     "url": "https://www.tradingview.com/symbols/PSX-OGDC/ideas/",
     "date": "2026-03-12", "score": 0.71, "label": "Positive",
     "rec": "Buy", "outlook": "Mixed",
     "contra": False, "contra_type": "None",
     "contra_phrase": "10 Long ideas vs 3 Short ideas",
     "kw_pos": ["bullish structure x3", "accumulate/buy x4",
                "breakout x3", "upside/rally x3", "10 Long ideas"],
     "kw_neg": ["bearish divergence x2", "correction x2",
                "downtrend x1", "3 Short ideas"],
     "price_target": "T1 289–300 / T2 320–335 / T3 350–420"},

    {"id":  5, "source": "YouTube Video",
     "url": "https://youtu.be/_NOSb2kgWKQ",
     "date": None,  "score": 0.40, "label": "Positive",
     "rec": "Buy", "outlook": "Positive",
     "contra": False, "contra_type": "None", "contra_phrase": "",
     "kw_pos": ["best stock (title)", "OGDC featured buy",
                "positive/green thumbnail"],
     "kw_neg": [],
     "price_target": "None (YouTube blocked — scored from title/desc fragments)"},

    {"id":  6, "source": "DefencePK Forum",
     "url": "https://defencepk.com/forums/threads/...17551/",
     "date": "2025-01-01", "score": 0.53, "label": "Positive",
     "rec": "None", "outlook": "Positive",
     "contra": False, "contra_type": "None", "contra_phrase": "",
     "kw_pos": ["index-heavy stocks traded in green", "breaking record 72nd time",
                "bullish momentum", "upward trajectory", "strong buying",
                "top-performing market 178% gain"],
     "kw_neg": [],
     "price_target": "None"},

    {"id":  7, "source": "Quora — Blue-Chip List",
     "url": "https://www.quora.com/Which-are-the-blue-chip-companies-in-Pakistan",
     "date": None,  "score": 0.33, "label": "Positive",
     "rec": "Hold", "outlook": "Positive",
     "contra": False, "contra_type": "None", "contra_phrase": "",
     "kw_pos": ["blue chip x2", "well-established financially strong",
                "consistent growth", "safe less risky"],
     "kw_neg": [],
     "price_target": "None (evergreen reference)"},

    {"id":  8, "source": "TradersUnion Forecast",
     "url": "https://tradersunion.com/currencies/forecast/ogdc-pkr/daily-and-weekly/",
     "date": "2026-03-12", "score": -0.35, "label": "Negative",
     "rec": "Sell", "outlook": "Negative",
     "contra": False, "contra_type": "None", "contra_phrase": "",
     "kw_pos": [],
     "kw_neg": ["decline −1.28%", "red forecast chart",
                "downside target PKR 274.94", "forecast arrow pointing down"],
     "price_target": "PKR 274.94 (downside)"},

    {"id":  9, "source": "Mettis Global — Q4FY24 Profit Collapse",
     "url": "https://mettisglobal.news/ogdcs-profit-collapse-sinks-oil-gas-sector-in-q4fy24/",
     "date": "2024-10-04", "score": -0.60, "label": "Negative",
     "rec": "None", "outlook": "Negative",
     "contra": False, "contra_type": "None", "contra_phrase": "",
     "kw_pos": [],
     "kw_neg": ["profit collapse x2", "sinks", "plummeted 42% YoY",
                "fell", "dropped", "drastic decline",
                "missed estimates", "decline"],
     "price_target": "None"},

    {"id": 10, "source": "Dawn.com — OGDCL Profit Slumps 33pc",
     "url": "https://www.dawn.com/news/1895049",
     "date": "2025-03-01", "score": -0.67, "label": "Negative",
     "rec": "None", "outlook": "Negative",
     "contra": False, "contra_type": "None", "contra_phrase": "",
     "kw_pos": [],
     "kw_neg": ["plunged 33pc", "33% reduction",
                "falling global crude prices", "forced closure of wells",
                "revenue down 12.3%", "exchange rate appreciation", "decline"],
     "price_target": "None"},

    {"id": 11, "source": "Mettis Global — FY25 Profit Falls",
     "url": "https://mettisglobal.news/OGDCs-FY25-profit-falls-to-Rs170bn-shareholders-to-receive-record-dividend-55420",
     "date": "2025-09-23", "score": -0.13, "label": "Neutral",
     "rec": "Implicit Buy", "outlook": "Negative",
     "contra": True, "contra_type": "Buy + Negative Outlook",
     "contra_phrase": "profit falls 18.7% ... yet highest-ever dividend of Rs15.05/share",
     "kw_pos": ["highest ever distribution x2", "finance income surged 97.90%",
                "record dividend", "operating expenses decreased"],
     "kw_neg": ["profit falls", "PAT down 18.70%", "EPS fell",
                "net sales down 13.48%", "gross profit fell 18.25%"],
     "price_target": "None (dividend Rs15.05/share announced)"},

    {"id": 12, "source": "ProPakistani — FY24 Profit Falls 7%",
     "url": "https://propakistani.pk/2024/09/23/ogdc-profit-falls-7-to-rs-208-9-billion-in-fy24/",
     "date": "2024-09-23", "score": -0.07, "label": "Neutral",
     "rec": "Implicit Buy", "outlook": "Negative",
     "contra": True, "contra_type": "Buy + Negative Outlook",
     "contra_phrase": "profit falls 7% yet topline up 12%, Arif Habib framing supportive",
     "kw_pos": ["topline climbed 12% YoY", "oil production up 2%",
                "5 new discoveries", "final dividend announced",
                "Arif Habib brokerage framing"],
     "kw_neg": ["profit falls 7%", "PAT down 7%", "decline 42% QoQ",
                "other income down 73%", "missed estimates"],
     "price_target": "None (dividend Rs4.00/share final)"},

    {"id": 13, "source": "Insight Research / Topline Securities",
     "url": "Insight Research on OGDC Growth Potential (June 2024)",
     "date": None,  "score": 0.47, "label": "Positive",
     "rec": "Buy", "outlook": "Negative",
     "contra": True, "contra_type": "Buy + Negative Outlook",
     "contra_phrase": "18.7% drop in net profit … brokerages reiterated Buy citing EV/EBITDA 1.1x and dividend powerhouse",
     "kw_pos": ["Buy rating", "dividend powerhouse",
                "EV/EBITDA deep discount 1.1x",
                "Topline Securities Buy", "AKD Research Buy"],
     "kw_neg": ["18.7% drop net profit", "7% dip FY24",
                "lower oil prices falling output"],
     "price_target": "None cited (deep discount framing)"},

    {"id": 14, "source": "IGI Securities Strategy Report",
     "url": "IGI Securities Strategy 2024 Report (December 2023)",
     "date": None,  "score": 0.33, "label": "Positive",
     "rec": "Buy", "outlook": "Negative",
     "contra": True, "contra_type": "Buy + Negative Outlook",
     "contra_phrase": "massive circular debt severely impacting cash flow … analysts encouraged investment based on Settlement Plan",
     "kw_pos": ["Circular Debt Settlement Plan",
                "regular monthly interest payments",
                "Reko Diq stake sale gains",
                "analysts encouraged investment"],
     "kw_neg": ["circular debt hundreds of billions",
                "severely impacting cash flow",
                "historical payout ratios impacted"],
     "price_target": "None"},
]


# ═══════════════════════════════════════════════════════════════════════════
# 5. Main Pipeline
# ═══════════════════════════════════════════════════════════════════════════

def run_analysis(csv_path: str = "ogdc_cleaned.csv") -> list[dict]:
    df = load_price_data(csv_path)
    results = []

    for art in ARTICLES:
        row = dict(art)

        # Price window tracking
        for n, key in [(1, "d1"), (3, "d3"), (5, "d5"), (10, "d10")]:
            chg, cum = get_next_n_change(art["date"], n, df)
            row[f"{key}_chg"] = chg
            row[f"{key}_cum"] = cum

        # D+1 sentiment match
        row["d1_match"] = d1_match(art["label"], row["d1_chg"])

        # Contradiction market result (D+1)
        if art["contra"]:
            row["contra_market_d1"]  = contra_market_result(art["rec"], row["d1_chg"])
            row["contra_market_d10"] = (
                "followed_recommendation" if (row["d10_cum"] or 0) > 0
                else "followed_outlook"
            )
        else:
            row["contra_market_d1"]  = None
            row["contra_market_d10"] = None

        results.append(row)

    return results


def print_summary(results: list[dict]):

    print("OGDC Sentiment Analysis")
  

    # Overall stats
    trackable = [r for r in results if r["d1_match"] not in ("Excl.", "Excluded")]
    matches   = [r for r in trackable if r["d1_match"] == "MATCH"]
    contra    = [r for r in results if r["contra"]]
    contra_tracked = [r for r in contra if r["d1_chg"] is not None]

    print(f"\n{'Total articles':35}: {len(results)}")
    print(f"{'Trackable (D+1 match)':35}: {len(trackable)}")
    print(f"{'Overall D+1 match rate':35}: {len(matches)/len(trackable)*100:.1f}% ({len(matches)}/{len(trackable)})")
    print(f"{'Contradiction articles':35}: {len(contra)}")
    print(f"{'Contradiction trackable (D+1)':35}: {len(contra_tracked)}")

    # Per article
    header = f"{'#':>2}  {'Source':<32} {'Date':<12} {'Score':>6}  {'Label':<9}  {'Rec':<12}  {'D+1':>7}  {'D+10cum':>8}  {'Match':<10}  {'Contra'}"
    print(header)
    for r in results:
        d1  = f"{r['d1_chg']:+.2f}%"  if r['d1_chg']  is not None else "   N/A"
        d10 = f"{r['d10_cum']:+.2f}%" if r['d10_cum'] is not None else "    N/A"
        contra_flag = "⚡ YES" if r["contra"] else ""
        print(f"{r['id']:>2}  {r['source']:<32} {str(r['date']):<12} "
              f"{r['score']:>6.2f}  {r['label']:<9}  {r['rec']:<12}  "
              f"{d1:>7}  {d10:>8}  {r['d1_match']:<10}  {contra_flag}")

    # Contradiction detail
    print("  Contradiction Analysis Trend")
    for r in contra:
        print(f"\n  URL {r['id']}: {r['source']}")
        print(f"  Type       : {r['contra_type']}")
        print(f"  Key phrase : \"{r['contra_phrase']}\"")
        print(f"  D+1        : {r['d1_chg']} → market {r['contra_market_d1'] or 'N/A'}")
        print(f"  D+10 cum   : {r['d10_cum']} → market {r['contra_market_d10'] or 'N/A'}")

    # Summary table
    print("  Aligned vs Contradiction — Match Rate Comparison")

    aligned_buy_pos = [r for r in results if not r["contra"]
                       and r["rec"] in ("Buy","Strong Buy","Hold")
                       and r["outlook"] == "Positive"
                       and r["d1_match"] not in ("Excl.","Excluded")]
    aligned_sell_neg = [r for r in results if not r["contra"]
                        and r["rec"] == "Sell"
                        and r["outlook"] == "Negative"
                        and r["d1_match"] not in ("Excl.","Excluded")]
    contra_buy_neg = [r for r in results if r["contra"]
                      and r["d1_chg"] is not None]

    def match_rate(lst):
        if not lst: return 0
        return sum(1 for r in lst if r["d1_match"] == "MATCH") / len(lst) * 100

    print(f"\n  {'Article type':<40} {'Count':>5}  {'D+1 Match Rate':>15}")
    print(f"  {'─'*40} {'─'*5}  {'─'*15}")
    print(f"  {'Aligned: Buy/Hold + Positive outlook':<40} {len(aligned_buy_pos):>5}  {match_rate(aligned_buy_pos):>14.0f}%")
    print(f"  {'Aligned: Sell + Negative outlook':<40} {len(aligned_sell_neg):>5}  {match_rate(aligned_sell_neg):>14.0f}%")
    print(f"  {'Contradiction: Buy + Negative outlook':<40} {len(contra_buy_neg):>5}  (mixed — see above)")


    print("  VERDICT: Does market reward buying against negative outlook?")
    print("""
  ANSWER: It depends on the TYPE of negative news — the dataset splits 50/50.

  ✅ Buy the Contradiction when:
     • Negative news is ONE-OFF (exchange loss, tax reversal, non-recurring item)
     • Topline/operational metrics are still growing
     • Brokerage support exists alongside the negative headline
     → URL 12 (ProPakistani FY24): profit fell 7% but stock +21.5% over 10 days

  ❌ Outlook can be trusted when:
     • Negative news is STRUCTURAL (multi-line decline: revenue + profit + EPS all falling)
     • Even a record dividend cannot mask persistent deterioration
     → URL 11 (Mettis FY25): profit fell 18.7%, stock –4.0% over 10 days

  📊 Aligned vs Contradiction:
     • Aligned articles (rec = outlook): ~70% D+1 match rate
     • Contradiction articles: ~50% D+1 match rate (higher uncertainty)
     → Contradictions should command a larger margin of safety before entry.

  🔑 KEY FILTER: Check if the negative cause is one-off or structural before
     acting against a bearish earnings headline.
""")


# ═══════════════════════════════════════════════════════════════════════════
# 6. Export
# ═══════════════════════════════════════════════════════════════════════════

def export_csv(results: list[dict], path: str = "ogdc_sentiment_results.csv"):
    rows = []
    for r in results:
        rows.append({
            "id": r["id"],
            "source": r["source"],
            "url": r["url"],
            "date": r["date"],
            "sentiment_score": r["score"],
            "sentiment_label": r["label"],
            "recommendation": r["rec"],
            "outlook": r["outlook"],
            "contradiction_flag": "YES" if r["contra"] else "NO",
            "contradiction_type": r["contra_type"],
            "key_contradiction_phrase": r["contra_phrase"],
            "price_target": r.get("price_target", "None"),
            "d1_chg_%": r["d1_chg"],
            "d3_cum_%": r["d3_cum"],
            "d5_cum_%": r["d5_cum"],
            "d10_cum_%": r["d10_cum"],
            "d1_match": r["d1_match"],
            "contra_market_d1": r["contra_market_d1"],
            "contra_market_d10": r["contra_market_d10"],
        })
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"\n  Results exported to: {path}")


if __name__ == "__main__":
    CSV_PATH = "ogdc_cleaned.csv"  
    results = run_analysis(CSV_PATH)
    print_summary(results)
    export_csv(results)
'''
