"""
ogdc_backtest.py
================
Backtesting engine for Bollinger Band strategies on OGDC stock.

Implements:
  - Benchmark: Buy & Hold, SMA-50 Crossover
  - Strategy 1: BB Mean Reversion
  - Strategy 2: BB Breakout
  - Strategy 3: %B Momentum
  - Strategy 4: Squeeze Breakout

Metrics: Total Return, CAGR, Sharpe, Max DD, Calmar, Win Rate, Profit Factor
Outputs: 5 PNG charts + ogdc_backtest_results.csv
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

# paths for input data, outputs, and images
BASE = os.path.dirname(os.path.abspath(__file__))  # scripts folder
DATA_DIR = os.path.join(BASE, "..", "data", "processed")
os.makedirs(DATA_DIR, exist_ok=True)
DATA = os.path.join(DATA_DIR, "ogdc_trend_features.csv")  # input from trend analysis
OUT_CSV = os.path.join(DATA_DIR, "ogdc_backtest_results.csv")  # save results here
IMG_DIR = os.path.join(BASE, "..", "outputs", "images")
os.makedirs(IMG_DIR, exist_ok=True)
IMG_PRE = os.path.join(IMG_DIR, "bt_")  # prefix for backtest images

print(f"reading from: {DATA}")
print(f"file exists: {os.path.exists(DATA)}")

# charts with dark theme
plt.rcParams.update({
    "figure.facecolor":"#0d1117","axes.facecolor":"#161b22",
    "axes.edgecolor":"#30363d","axes.labelcolor":"#e6edf3",
    "xtick.color":"#8b949e","ytick.color":"#8b949e",
    "text.color":"#e6edf3","grid.color":"#21262d","grid.linewidth":0.6,
    "legend.facecolor":"#161b22","legend.edgecolor":"#30363d",
})
C = {"price":"#58a6ff","eq1":"#3fb950","eq2":"#f0883e","eq3":"#d2a8ff",
     "eq4":"#ffa657","bh":"#8b949e","sma":"#79c0ff",
     "up":"#2ea043","dn":"#f85149","buy":"#2ea043","sell":"#f85149",
     "dd":"#f85149","neutral":"#8b949e"}

TRADING_DAYS = 252  # annualisation factor for sharpe ratio
COST         = 0.0017    # 0.17% round-trip brokerage (realistic for psx)
STOP_LOSS    = 0.03      # 3% hard stop per position
TRAIN_SPLIT  = 0.70      # 70% in-sample, 30% out-of-sample

# save figure helper
def save_fig(fig, name):
    path = IMG_PRE + name
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  → saved: bt_{name}")

# console formatting helpers
def banner(t): print(f"\n{'═'*70}\n  {t}\n{'═'*70}")
def sub(t):    print(f"\n  {'─'*58}\n  {t}\n  {'─'*58}")

# ══════════════════════════════════════════════════════════════════════════════
# load data
# ══════════════════════════════════════════════════════════════════════════════
banner("loading data")
df = pd.read_csv(DATA, index_col="Date", parse_dates=True)
df.sort_index(inplace=True)

# drop rows with missing values in core columns needed for backtest
core = ["Price","BB_upper","BB_lower","BB_mid","BB_pct_b","BB_bandwidth","BB_squeeze","SMA_50","Returns"]
df.dropna(subset=core, inplace=True)

price = df["Price"]
ret   = df["Returns"] / 100   # convert percentage to fractional daily return

train_end = df.index[int(len(df) * TRAIN_SPLIT)]  # chronological split point
print(f"  rows         : {len(df)}")
print(f"  full period  : {df.index.min().date()} → {df.index.max().date()}")
print(f"  train        : {df.index.min().date()} → {train_end.date()}")
print(f"  test (oos)   : {(train_end + pd.Timedelta(days=1)).date()} → {df.index.max().date()}")
print(f"  cost/trade   : {COST*100:.2f}%  |  stop-loss: {STOP_LOSS*100:.0f}%")

# ══════════════════════════════════════════════════════════════════════════════
# performance metrics (sharpe, drawdown, calmar, win rate, profit factor)
# ══════════════════════════════════════════════════════════════════════════════
def compute_metrics(equity: pd.Series, trades: pd.DataFrame, label: str) -> dict:
    """calculate risk-adjusted returns and trade statistics"""
    eq = equity.dropna()
    if len(eq) < 2:
        return {}
    n_years   = len(eq) / TRADING_DAYS
    total_ret = (eq.iloc[-1] / eq.iloc[0] - 1) * 100  # cumulative return %
    cagr      = ((eq.iloc[-1] / eq.iloc[0]) ** (1/max(n_years,0.01)) - 1) * 100  # annualised
    daily_ret = eq.pct_change().dropna()
    sharpe    = (daily_ret.mean() / daily_ret.std() * np.sqrt(TRADING_DAYS) if daily_ret.std() > 0 else 0)  # risk-adjusted return
    rolling_max  = eq.cummax()
    drawdown     = (eq - rolling_max) / rolling_max  # peak-to-trough decline
    max_dd       = drawdown.min() * 100  # worst drawdown %
    calmar        = cagr / abs(max_dd) if max_dd != 0 else 0  # return per unit of max risk

    # trade-level statistics (win rate, profit factor)
    if len(trades) > 0:
        wins      = trades[trades["pnl_pct"] > 0]
        losses    = trades[trades["pnl_pct"] <= 0]
        win_rate  = len(wins) / len(trades) * 100  # % of winning trades
        gross_win = wins["pnl_pct"].sum()
        gross_los = abs(losses["pnl_pct"].sum())
        pf        = gross_win / gross_los if gross_los > 0 else np.inf  # gross profit / gross loss
        n_trades  = len(trades)
    else:
        win_rate = pf = n_trades = 0

    return {
        "Strategy":   label,
        "Total_Ret%": round(total_ret, 2),
        "CAGR%":      round(cagr, 2),
        "Sharpe":     round(sharpe, 3),
        "MaxDD%":     round(max_dd, 2),
        "Calmar":     round(calmar, 3),
        "WinRate%":   round(win_rate, 1),
        "ProfitFactor": round(pf, 2) if pf != np.inf else "∞",
        "N_Trades":   n_trades,
    }

# ══════════════════════════════════════════════════════════════════════════════
# backtesting engine with transaction costs and stop-loss
# ══════════════════════════════════════════════════════════════════════════════
def run_backtest(signal_series: pd.Series, price: pd.Series,
                 ret: pd.Series, label: str,
                 stop_loss: float = STOP_LOSS,
                 cost: float = COST) -> tuple:
    """
    simulate trading with signals, costs, and hard stop-loss.
    signal_series: 1=long, 0=flat (long-only strategy)
    returns equity curve and trade log
    """
    equity   = [1.0]  # start with 100% capital
    position = 0        # 0 = flat, 1 = long
    entry_price = None
    trade_log   = []
    idx = price.index

    for i in range(1, len(idx)):
        sig     = int(signal_series.iloc[i-1])   # use prior bar's signal (no lookahead bias)
        p_today = price.iloc[i]
        r_today = ret.iloc[i]
        eq_prev = equity[-1]

        # stop-loss check (exit if price drops 3% from entry)
        if position == 1 and entry_price is not None:
            if (p_today - entry_price) / entry_price <= -stop_loss:
                pnl = -stop_loss - cost  # stop loss hit + exit cost
                trade_log.append({"date": idx[i], "type": "stop", "pnl_pct": pnl})
                equity.append(eq_prev * (1 + pnl))
                position = 0
                entry_price = None
                continue

        # signal transitions
        if sig == 1 and position == 0:
            # enter long position
            position    = 1
            entry_price = p_today
            equity.append(eq_prev * (1 - cost))   # pay entry brokerage

        elif sig == 0 and position == 1:
            # exit long position
            entry_p = entry_price if entry_price else p_today
            pnl = (p_today - entry_p) / entry_p - cost
            trade_log.append({"date": idx[i], "type": "exit", "pnl_pct": pnl})
            equity.append(eq_prev * (1 + r_today - cost))
            position    = 0
            entry_price = None

        elif position == 1:
            # hold existing position
            equity.append(eq_prev * (1 + r_today))
        else:
            # stay flat (cash)
            equity.append(eq_prev)

    eq_series = pd.Series(equity, index=idx[:len(equity)])
    trades_df = pd.DataFrame(trade_log) if trade_log else pd.DataFrame(columns=["date","type","pnl_pct"])
    return eq_series, trades_df

# ══════════════════════════════════════════════════════════════════════════════
# strategy definitions
# ══════════════════════════════════════════════════════════════════════════════
banner("running strategies")

all_equity   = {}
all_trades   = {}
all_signals  = {}
all_metrics  = []

# benchmark 1: buy and hold (passive investment)
sub("benchmark: buy and hold")
bh_equity = price / price.iloc[0]
bh_ret    = ret.copy()
bh_trades = pd.DataFrame(columns=["date","type","pnl_pct"])
all_equity["buy & hold"] = bh_equity
all_metrics.append(compute_metrics(bh_equity, bh_trades, "buy & hold"))
print(f"  total return: {(bh_equity.iloc[-1]-1)*100:.1f}%")

# benchmark 2: sma-50 crossover (trend following)
sub("benchmark: sma-50 crossover")
sma_sig = (price > df["SMA_50"]).astype(int)  # 1 when price above 50-day ma
sma_eq, sma_trades = run_backtest(sma_sig, price, ret, "sma-50 cross")
all_equity["sma-50 cross"] = sma_eq
all_trades["sma-50 cross"] = sma_trades
m = compute_metrics(sma_eq, sma_trades, "sma-50 cross")
all_metrics.append(m)
print(f"  trades: {m['N_Trades']}  |  total return: {m['Total_Ret%']}%  |  sharpe: {m['Sharpe']}")

# strategy 1: bb mean reversion (buy near lower band, sell at middle)
sub("strategy 1: bollinger band mean reversion")
# buy when %b < 0.05 (price near/below lower band), exit when %b > 0.50 (price reaches middle)
pct_b = df["BB_pct_b"].fillna(0.5)
mr_sig = pd.Series(0, index=df.index)
in_pos = False
for i in range(1, len(df)):
    if not in_pos and pct_b.iloc[i] < 0.05:
        in_pos = True
    elif in_pos and pct_b.iloc[i] > 0.50:
        in_pos = False
    mr_sig.iloc[i] = int(in_pos)

mr_eq, mr_trades = run_backtest(mr_sig, price, ret, "bb mean reversion")
all_equity["bb mr"]    = mr_eq
all_trades["bb mr"]    = mr_trades
all_signals["bb mr"]   = mr_sig
m = compute_metrics(mr_eq, mr_trades, "bb mean reversion")
all_metrics.append(m)
print(f"  trades: {m['N_Trades']}  |  total return: {m['Total_Ret%']}%  |  sharpe: {m['Sharpe']}")
print(f"  win rate: {m['WinRate%']}%  |  max dd: {m['MaxDD%']}%  |  calmar: {m['Calmar']}")

# strategy 2: bb breakout (momentum following)
sub("strategy 2: bollinger band breakout")
# buy when price closes above upper band, exit when price falls below middle band
bo_sig = pd.Series(0, index=df.index)
in_pos = False
for i in range(1, len(df)):
    if not in_pos and price.iloc[i] > df["BB_upper"].iloc[i]:
        in_pos = True
    elif in_pos and price.iloc[i] < df["BB_mid"].iloc[i]:
        in_pos = False
    bo_sig.iloc[i] = int(in_pos)

bo_eq, bo_trades = run_backtest(bo_sig, price, ret, "bb breakout")
all_equity["bb breakout"]  = bo_eq
all_trades["bb breakout"]  = bo_trades
all_signals["bb breakout"] = bo_sig
m = compute_metrics(bo_eq, bo_trades, "bb breakout")
all_metrics.append(m)
print(f"  trades: {m['N_Trades']}  |  total return: {m['Total_Ret%']}%  |  sharpe: {m['Sharpe']}")
print(f"  win rate: {m['WinRate%']}%  |  max dd: {m['MaxDD%']}%  |  calmar: {m['Calmar']}")

# strategy 3: %b momentum (cross above/below 0.5)
sub("strategy 3: %b momentum")
# long when %b crosses above 0.5 (momentum), exit when crosses below 0.5
pbm_sig = pd.Series(0, index=df.index)
in_pos = False
for i in range(1, len(df)):
    pb_prev = pct_b.iloc[i-1]
    pb_curr = pct_b.iloc[i]
    if not in_pos and pb_prev <= 0.5 and pb_curr > 0.5:
        in_pos = True
    elif in_pos and pb_prev >= 0.5 and pb_curr < 0.5:
        in_pos = False
    pbm_sig.iloc[i] = int(in_pos)

pbm_eq, pbm_trades = run_backtest(pbm_sig, price, ret, "%b momentum")
all_equity["%b momentum"]  = pbm_eq
all_trades["%b momentum"]  = pbm_trades
all_signals["%b momentum"] = pbm_sig
m = compute_metrics(pbm_eq, pbm_trades, "%b momentum")
all_metrics.append(m)
print(f"  trades: {m['N_Trades']}  |  total return: {m['Total_Ret%']}%  |  sharpe: {m['Sharpe']}")
print(f"  win rate: {m['WinRate%']}%  |  max dd: {m['MaxDD%']}%  |  calmar: {m['Calmar']}")

# strategy 4: squeeze breakout (trade direction after band contraction)
sub("strategy 4: bb squeeze breakout")
'''
squeeze days in stocks is a rapid, dramatic spike in a stock’s price that forces 
investors who bet against it to panic-buy shares to limit their losses
'''
# after squeeze (bandwidth < 20th percentile), trade direction of first breakout
squeeze = df["BB_squeeze"].fillna(False)
sq_sig  = pd.Series(0, index=df.index)
in_pos  = False
in_sq   = False

for i in range(1, len(df)):
    was_sq = in_sq
    in_sq  = bool(squeeze.iloc[i])
    p      = price.iloc[i]

    if not in_pos:
        if was_sq and not in_sq:  # squeeze just ended
            if p > df["BB_upper"].iloc[i]:  # bullish breakout
                in_pos = True
    else:
        if p < df["BB_mid"].iloc[i]:  # exit at middle band
            in_pos = False
    sq_sig.iloc[i] = int(in_pos)

sq_eq, sq_trades = run_backtest(sq_sig, price, ret, "squeeze breakout")
all_equity["squeeze bo"]  = sq_eq
all_trades["squeeze bo"]  = sq_trades
all_signals["squeeze bo"] = sq_sig
m = compute_metrics(sq_eq, sq_trades, "squeeze breakout")
all_metrics.append(m)
print(f"  trades: {m['N_Trades']}  |  total return: {m['Total_Ret%']}%  |  sharpe: {m['Sharpe']}")
print(f"  win rate: {m['WinRate%']}%  |  max dd: {m['MaxDD%']}%  |  calmar: {m['Calmar']}")

# ══════════════════════════════════════════════════════════════════════════════
# walk-forward out-of-sample results (last 30% of data)
# ══════════════════════════════════════════════════════════════════════════════
banner("walk-forward out-of-sample results")
'''
testing optimized parameters on unseen data across multiple, sequential, non-overlapping periods.
'''

oos_metrics = []
oos_equity  = {}

for name, sig in all_signals.items():
    sig_oos   = sig[sig.index > train_end]
    price_oos = price[price.index > train_end]
    ret_oos   = ret[ret.index > train_end]
    if len(sig_oos) < 10:
        continue
    eq_oos, tr_oos = run_backtest(sig_oos, price_oos, ret_oos, name)
    oos_equity[name] = eq_oos
    m = compute_metrics(eq_oos, tr_oos, f"{name} [oos]")
    oos_metrics.append(m)
    print(f"  {name:20s}: return={m['Total_Ret%']:7.2f}%  "
          f"sharpe={m['Sharpe']:6.3f}  maxdd={m['MaxDD%']:7.2f}%  "
          f"winrate={m['WinRate%']:5.1f}%")

# buy-and-hold oos benchmark
bh_oos = price[price.index > train_end]
bh_oos = bh_oos / bh_oos.iloc[0]
oos_equity["buy & hold"] = bh_oos
m_bh_oos = compute_metrics(bh_oos, pd.DataFrame(), "buy & hold [oos]")
oos_metrics.append(m_bh_oos)
print(f"  {'buy & hold':20s}: return={m_bh_oos['Total_Ret%']:7.2f}%  sharpe={m_bh_oos['Sharpe']:6.3f}  maxdd={m_bh_oos['MaxDD%']:7.2f}%")

# ══════════════════════════════════════════════════════════════════════════════
# summary tables
# ══════════════════════════════════════════════════════════════════════════════
banner("full results table")
results_df = pd.DataFrame(all_metrics)
print(f"\n  in-sample (full period):")
print(results_df.to_string(index=False))

oos_df = pd.DataFrame(oos_metrics)
print(f"\n  out-of-sample (last 30%):")
print(oos_df.to_string(index=False))

best_is  = results_df.iloc[2:].loc[results_df.iloc[2:]["Sharpe"].idxmax(), "Strategy"]
best_oos = oos_df.loc[oos_df["Sharpe"].idxmax(), "Strategy"]
print(f"\n  best is strategy (sharpe): {best_is}")
print(f"  best oos strategy (sharpe): {best_oos}")

# save results
results_df.to_csv(OUT_CSV, index=False)

# ══════════════════════════════════════════════════════════════════════════════
# visualisations
# ══════════════════════════════════════════════════════════════════════════════
banner("generating charts")

strat_colors = {
    "buy & hold":   C["bh"],
    "sma-50 cross": C["sma"],
    "bb mr":        C["eq1"],
    "bb breakout":  C["eq2"],
    "%b momentum":  C["eq3"],
    "squeeze bo":   C["eq4"],
}

# chart 1: full equity curves with drawdown panel
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
fig.subplots_adjust(hspace=0.08)

for name, eq in all_equity.items():
    lw = 1.8 if name == "buy & hold" else 1.1
    ls = "--" if name in ["buy & hold","sma-50 cross"] else "-"
    ax1.plot(eq.index, eq * 100, lw=lw, linestyle=ls,
             color=strat_colors.get(name, "#8b949e"), label=name)

ax1.axvline(train_end, color="white", lw=1, linestyle=":", alpha=0.6)
ax1.text(train_end, ax1.get_ylim()[0] if ax1.get_ylim()[0] != 0 else 50,
         "  oos →", color="white", fontsize=8, va="bottom", alpha=0.7)
ax1.set_ylabel("equity (base=100)", fontsize=10)
ax1.set_title("ogdc — bollinger band strategies: equity curves (full period)",
               fontsize=13, fontweight="bold", pad=10)
ax1.legend(fontsize=9, ncol=3)
ax1.grid(True, alpha=0.25)
ax1.set_yscale("log")
ax1.yaxis.set_major_formatter(plt.ScalarFormatter())

# drawdown panel (peak-to-trough declines)
for name, eq in all_equity.items():
    dd = (eq / eq.cummax() - 1) * 100
    ax2.plot(dd.index, dd, lw=0.8, color=strat_colors.get(name, "#8b949e"),
             alpha=0.75, label=name)
ax2.axhline(0, color="#8b949e", lw=0.7)
ax2.fill_between(all_equity["buy & hold"].index,
                 (all_equity["buy & hold"] / all_equity["buy & hold"].cummax() - 1)*100,
                 0, alpha=0.1, color=C["bh"])
ax2.axvline(train_end, color="white", lw=1, linestyle=":", alpha=0.6)
ax2.set_ylabel("drawdown %", fontsize=10)
ax2.set_xlabel("date", fontsize=9)
ax2.grid(True, alpha=0.25)
save_fig(fig, "01_equity_curves.png")

# chart 2: out-of-sample only equity curves
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 9), sharex=True)
fig.subplots_adjust(hspace=0.08)

for name, eq in oos_equity.items():
    lw = 1.8 if name == "buy & hold" else 1.1
    ls = "--" if name == "buy & hold" else "-"
    ax1.plot(eq.index, eq * 100, lw=lw, linestyle=ls,
             color=strat_colors.get(name, "#8b949e"), label=name)

ax1.set_ylabel("equity (base=100)", fontsize=10)
ax1.set_title("ogdc — out-of-sample performance only",
               fontsize=13, fontweight="bold", pad=10)
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.25)

# drawdown panel for oos period
for name, eq in oos_equity.items():
    dd = (eq / eq.cummax() - 1) * 100
    ax2.plot(dd.index, dd, lw=0.8, color=strat_colors.get(name, "#8b949e"), alpha=0.75)
ax2.axhline(0, color="#8b949e", lw=0.7)
ax2.set_ylabel("drawdown %", fontsize=10)
ax2.set_xlabel("date", fontsize=9)
ax2.grid(True, alpha=0.25)
save_fig(fig, "02_oos_curves.png")

# chart 3: strategy performance bar chart (return, sharpe, drawdown, win rate)
metrics_plot = results_df[results_df["Strategy"].isin(
    ["bb mean reversion","bb breakout","%b momentum","squeeze breakout","buy & hold","sma-50 cross"]
)]
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.subplots_adjust(hspace=0.4, wspace=0.3)

metrics_cols = ["Total_Ret%","Sharpe","MaxDD%","WinRate%"]
titles       = ["total return (%)", "sharpe ratio", "max drawdown (%)", "win rate (%)"]

for ax, col, title in zip(axes.flat, metrics_cols, titles):
    vals   = metrics_plot[col].apply(lambda x: float(x) if x != "∞" else 0)
    names  = metrics_plot["Strategy"].str.replace(" ", "\n")
    colors = [strat_colors.get(s, "#8b949e") for s in metrics_plot["Strategy"]]
    bars   = ax.bar(range(len(names)), vals, color=colors, edgecolor="#0d1117", alpha=0.85)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=7.5)
    ax.set_title(title, fontweight="bold", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + abs(vals.max())*0.01,
                f"{v:.1f}", ha="center", va="bottom", fontsize=7.5, color=C["price"])

fig.suptitle("strategy performance comparison — ogdc", fontsize=13, fontweight="bold")
save_fig(fig, "03_performance_bars.png")

# chart 4: bb mean reversion trade visualisation (2024-2026 zoom)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 9), sharex=True)
fig.subplots_adjust(hspace=0.08)

zoom = df["2024":]
zp   = price["2024":]

# bollinger bands with shaded area
ax1.fill_between(zoom.index, zoom["BB_upper"], zoom["BB_lower"],
                 color="#1f4080", alpha=0.2)
ax1.plot(zoom.index, zoom["BB_upper"], lw=1,   color="#388bfd", linestyle="--", alpha=0.7)
ax1.plot(zoom.index, zoom["BB_mid"],   lw=0.9, color="#8b949e", alpha=0.6)
ax1.plot(zoom.index, zoom["BB_lower"], lw=1,   color="#388bfd", linestyle="--", alpha=0.7)
ax1.plot(zp.index,  zp,               lw=1.2, color=C["price"], zorder=3, label="price")

# mark entry and exit signals
mr_sig_z = all_signals["bb mr"].reindex(zoom.index).fillna(0)
sig_change = mr_sig_z.diff()
entries = zoom.index[sig_change == 1]
exits   = zoom.index[sig_change == -1]

ax1.scatter(entries, zp.reindex(entries), marker="^", color=C["buy"],  s=80, zorder=5, label=f"buy ({len(entries)})")
ax1.scatter(exits,   zp.reindex(exits),   marker="v", color=C["sell"], s=80, zorder=5, label=f"sell ({len(exits)})")
ax1.fill_between(zoom.index, zoom["BB_lower"], zoom["BB_upper"],
                 where=mr_sig_z == 1, color=C["up"], alpha=0.08, label="in position")
ax1.set_title("bb mean-reversion strategy — trades (2024–2026)", fontsize=12, fontweight="bold")
ax1.set_ylabel("price (pkr)")
ax1.legend(fontsize=8, ncol=4)
ax1.grid(True, alpha=0.25)

# equity curve for the same period
eq_z = all_equity["bb mr"].reindex(zoom.index, method="ffill")
bh_z = all_equity["buy & hold"].reindex(zoom.index, method="ffill")
ax2.plot(eq_z.index, eq_z * 100, lw=1.2, color=C["eq1"], label="bb mr strategy")
ax2.plot(bh_z.index, bh_z * 100, lw=1.2, color=C["bh"],  linestyle="--", label="buy & hold")
ax2.set_ylabel("equity (base=100)")
ax2.set_xlabel("date")
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.25)
save_fig(fig, "04_mr_trades.png")

# chart 5: rolling 12-month sharpe ratio
fig, ax = plt.subplots(figsize=(16, 5))
window = TRADING_DAYS

for name, eq in all_equity.items():
    dr = eq.pct_change().dropna()
    roll_sharpe = dr.rolling(window).apply(
        lambda x: x.mean() / x.std() * np.sqrt(TRADING_DAYS) if x.std() > 0 else 0
    )
    lw = 1.6 if name == "buy & hold" else 1.0
    ls = "--" if name in ["buy & hold","sma-50 cross"] else "-"
    ax.plot(roll_sharpe.index, roll_sharpe,
            lw=lw, linestyle=ls, color=strat_colors.get(name, "#8b949e"),
            label=name, alpha=0.85)

ax.axhline(0, color="#8b949e", lw=0.8)
ax.axhline(1, color="#8b949e", lw=0.8, linestyle=":", alpha=0.5)
ax.axvline(train_end, color="white", lw=1, linestyle=":", alpha=0.6)
ax.fill_between(ax.get_xlim(), 0, 1, alpha=0.04, color="#8b949e")
ax.set_title("rolling 12-month sharpe ratio — all strategies", fontsize=13, fontweight="bold")
ax.set_ylabel("sharpe ratio")
ax.set_xlabel("date")
ax.legend(fontsize=9, ncol=3)
ax.grid(True, alpha=0.25)
save_fig(fig, "05_rolling_sharpe.png")

# ══════════════════════════════════════════════════════════════════════════════
# final summary
# ══════════════════════════════════════════════════════════════════════════════
banner("summary")
print(f"""
  ┌─────────────────────────────────────────────────────────────────┐
  │  bollinger band backtesting summary                             │
  ├────────────────────────┬────────────────────────────────────────┤
  │ transaction cost       │ {COST*100:.2f}% per trade                       │
  │ stop loss              │ {STOP_LOSS*100:.0f}% hard stop per position          │
  │ best is strategy       │ {best_is:<38} │
  │ best oos strategy      │ {best_oos:<38} │
  ├────────────────────────┴────────────────────────────────────────┤""")

for _, row in results_df.iterrows():
    print(f"  │  {row['Strategy']:22s} return={str(row['Total_Ret%']):>7}%  "
          f"sharpe={str(row['Sharpe']):>6}  dd={str(row['MaxDD%']):>7}%  │")

print(f"  └─────────────────────────────────────────────────────────────────┘")
print(f"\n  saved: ogdc_backtest_results.csv  +  5 charts")