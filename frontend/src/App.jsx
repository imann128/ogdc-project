import { useState, useEffect, useRef } from "react";
import PropTypes from "prop-types";
import {
  Line, AreaChart, Area, BarChart, Bar, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, ReferenceLine, ComposedChart,
} from "recharts";

// ── Inline price data (replace with import or fetch in production) ──────────
const PRICE_DATA_URL = "/price_data.json";

// ── Color system ─────────────────────────────────────────────────────────────
const C = {
  cyan:    "#00e5ff",
  cyanDim: "#00e5ff44",
  green:   "#00ff9d",
  amber:   "#ffb300",
  red:     "#ff3d5a",
  purple:  "#c084fc",
  bg0:     "#02060f",
  bg1:     "#060d1a",
  bg2:     "#0a1628",
  bg3:     "#0f2040",
  border:  "#00e5ff18",
  borderHover: "#00e5ff44",
  textPrimary: "#e2f0ff",
  textSecondary: "#7aa8cc",
  textMuted: "#3a6080",
};

// ── Chart sections catalogue ──────────────────────────────────────────────────
const SECTIONS = [
  {
    id: "eda",
    label: "Exploratory Analysis",
    icon: "◈",
    color: C.cyan,
    charts: [
      { file: "stat_01_eda_panel.png",     title: "EDA Dashboard Panel",
        desc: "Price trend, return distribution, annual boxplots, volume, Q-Q plot" },
      { file: "stat_02_scatter_matrix.png", title: "Scatter Matrix",
        desc: "Pairwise relationships: Price, Volume, Returns, HV-20 with Pearson r" },
    ],
    findings: [
      "Price rose from PKR 69.55 → 333.90 (+380%) over 6 years",
      "Daily returns: mean 0.075%, std 2.63%, excess kurtosis 41.7",
      "Extreme fat tails — ±5% daily moves occur far more than Normal predicts",
      "Price ~ Volume Spearman ρ=0.35 (p<0.001): bull phases attract more trading",
    ],
  },
  {
    id: "statistics",
    label: "Statistical Tests",
    icon: "∑",
    color: C.purple,
    charts: [
      { file: "stat_03_anova.png",               title: "ANOVA Analysis",
        desc: "One-way F-test by year + two-way Year × Volatility Regime interaction" },
      { file: "stat_04_distribution_fitting.png", title: "Distribution Fitting",
        desc: "MLE fit: Normal, Student-t, Laplace, Logistic, Cauchy vs returns histogram" },
      { file: "stat_05_regression.png",           title: "Multiple Regression Diagnostics",
        desc: "OLS: Price ~ Open + High + Low + Volume + Lag1 + Lag2, R²=0.9995" },
      { file: "stat_06_arima.png",                title: "ARIMA Time Series",
        desc: "ACF/PACF model identification, ARMA(0,3) fit, 30-day forecast" },
      { file: "stat_07_nonparametric.png",        title: "Nonparametric Tests",
        desc: "Kruskal-Wallis, Mann-Whitney U, Spearman correlation, Runs Test" },
    ],
    findings: [
      "ANOVA F=0.852 (p=0.530) — no significant year-on-year return difference",
      "Student-t best fit (AIC=6,745 vs Normal 7,408 — gap of 663 points)",
      "Student-t ν≈2.8 df: tail risk 14× higher than Normal model predicts",
      "ARMA(0,3) selected; Ljung-Box residuals confirm white noise at all lags",
      "Runs Test Z=0.043 (p=0.97) — return signs are random, supporting EMH",
    ],
  },
  {
    id: "ml",
    label: "ML Models",
    icon: "⬡",
    color: C.green,
    charts: [
      { file: "comparison_01_confusion_matrices.png", title: "Confusion Matrices",
        desc: "Side-by-side: Random Forest vs GBM Classifier on test set" },
      { file: "comparison_02_roc_curves.png",         title: "ROC Curves",
        desc: "Overlaid ROC curves with 95% bootstrap confidence intervals" },
      { file: "comparison_03_precision_recall_f1.png",title: "Precision / Recall / F1",
        desc: "Macro-averaged classification metrics comparison" },
      { file: "comparison_06_regime_timeseries.png",  title: "K-Means Regime Detection",
        desc: "OGDC price coloured by market regime (K=2, Silhouette=0.40)" },
      { file: "comparison_08_feature_importance.png", title: "Feature Importance",
        desc: "Top features for RF classifier, GBM classifier, GBM regressor" },
      { file: "ogdc_rf_feature_importance.png",       title: "RF Feature Importance (Detail)",
        desc: "Top 10 features driving Random Forest direction predictions" },
      { file: "ogdc_regimes.png",                     title: "Price Coloured by Regime",
        desc: "Full price history overlaid with K-Means regime assignments" },
      { file: "ogdc_kmeans_elbow.png",                title: "Elbow + Silhouette",
        desc: "Optimal K selection — Elbow method and Silhouette score" },
    ],
    findings: [
      "RF Classifier ROC-AUC = 0.9993, GBM = 0.9967 — both beat naive baseline",
      "K-Means K=2 optimal (Silhouette=0.40): bearish −0.61% avg / bullish +1.95%",
      "GBM Regressor R²=0.9955, Directional Accuracy=99.68% on test set",
      "RF and GBM agree on 100% of test predictions — ensembling adds no value",
      "Note: high accuracy partly driven by same-day ChangeP column (data leakage)",
    ],
  },
  {
    id: "garch",
    label: "GARCH Volatility",
    icon: "〜",
    color: C.amber,
    charts: [
      { file: "garch_01_all_models.png",         title: "All 4 GARCH Models",
        desc: "ARCH(1), GARCH(1,1), EGARCH(1,1), GJR-GARCH conditional volatilities" },
      { file: "garch_02_leverage_comparison.png", title: "Leverage Effect",
        desc: "Symmetric vs asymmetric models — EGARCH minus GARCH difference" },
      { file: "garch_03_forecast.png",            title: "30-Day Volatility Forecast",
        desc: "GARCH(1,1) forward forecast with regime-coloured bars and 95% CI" },
      { file: "garch_04_news_impact.png",         title: "News Impact Curves",
        desc: "How positive vs negative shocks of equal size affect next-period vol" },
      { file: "garch_05_diagnostics.png",         title: "Residual Diagnostics",
        desc: "Standardised residuals, distribution vs Normal/t, Q-Q plot" },
    ],
    findings: [
      "GJR-GARCH(1,1) wins by AIC — leverage effect γ=0.33 confirmed",
      "Negative shocks inflate future volatility 33% more than positive shocks",
      "Persistence α+β=0.78, shock half-life only 2.8 days (fast mean-reversion)",
      "Long-run unconditional volatility: 41.9% annualised",
      "ARCH effect Q(10)=226.96 (p≈0) — volatility clustering strongly confirmed",
    ],
  },
  {
    id: "trend",
    label: "Trend Analysis",
    icon: "◟",
    color: "#38bdf8",
    charts: [
      { file: "trend_01_moving_averages.png",    title: "Moving Averages & Crosses",
        desc: "SMA 20/50/200, EMA 12/26, WMA 20 with Golden/Death cross markers" },
      { file: "trend_02_bollinger_bands.png",    title: "Bollinger Bands (Full)",
        desc: "Classic 20-day 2σ BB with %B indicator and bandwidth panel" },
      { file: "trend_03_bb_zoom_2024.png",       title: "BB Zoom 2024–2026",
        desc: "Mean-reversion buy/sell signals highlighted on recent data" },
      { file: "trend_04_volatility_panel.png",   title: "Volatility Panel",
        desc: "HV-20/60/120, Parkinson, Garman-Klass estimators + regime bands" },
      { file: "trend_05_ts_decomposition.png",   title: "Time Series Decomposition",
        desc: "Additive decomposition: Trend (252-day MA), Seasonal, Residual" },
      { file: "trend_06_acf_pacf.png",           title: "ACF / PACF",
        desc: "Autocorrelation structure of OGDC daily returns up to lag 40" },
      { file: "trend_07_volatility_regimes.png", title: "Volatility Regimes",
        desc: "HV-20 over time coloured by regime, HV vs |Return| scatter" },
      { file: "trend_08_integrated_dashboard.png",title: "Integrated Dashboard",
        desc: "All indicators: Price + BB + SMA + %B + Bandwidth + HV + Returns" },
    ],
    findings: [
      "4 Golden Crosses, 3 Death Crosses detected — net bullish bias over period",
      "Price above SMA-200 on 61.7% of trading days — structural uptrend",
      "287 squeeze days (BB bandwidth < 20th pct) — compressed vol before big moves",
      "MR Buy signal: 56.9% next-day win rate, avg return +0.38% (best BB signal)",
      "Garman-Klass most efficient OHLC volatility estimator for OGDC",
    ],
  },
  {
    id: "backtest",
    label: "Backtesting",
    icon: "◎",
    color: "#4ade80",
    charts: [
      { file: "bt_01_equity_curves.png",     title: "Equity Curves (Full Period)",
        desc: "Log-scale equity + drawdown panel for all strategies vs benchmarks" },
      { file: "bt_02_oos_curves.png",        title: "Out-of-Sample Performance",
        desc: "Last 30% of data only — walk-forward validation" },
      { file: "bt_03_performance_bars.png",  title: "Performance Metrics",
        desc: "Total Return, Sharpe, Max Drawdown, Win Rate — all strategies" },
      { file: "bt_04_mr_trades.png",         title: "MR Trade Signals (2024–26)",
        desc: "Entry/exit signals on price chart with equity curve below" },
      { file: "bt_05_rolling_sharpe.png",    title: "Rolling 12-Month Sharpe",
        desc: "Regime-aware performance — shows when each strategy works" },
    ],
    findings: [
      "BB Mean Reversion: Sharpe=0.57, Max DD=−16.5% (vs −41% Buy & Hold)",
      "Same Sharpe as passive but half the drawdown — best risk-adjusted strategy",
      "BB Breakout best OOS (2024–26): Sharpe=1.04, Win Rate=66.7%",
      "OGDC shifted from mean-reverting to momentum regime after 2023",
      "Transaction cost 0.17% + 3% stop-loss applied — realistic PSX simulation",
    ],
  },
  {
    id: "sentiment",
    label: "Sentiment & Contradiction",
    icon: "◇",
    color: "#f472b6",
    charts: [
      { file: "contra_02_source_matrix.png",          title: "14-Source Signal Matrix",
        desc: "Heatmap: Sentiment, Rec, Outlook, Contradiction flag, Model accuracy" },
      { file: "contra_03_d10_returns_timeline.png",   title: "D+10 Returns Timeline",
        desc: "Post-publication returns — red = contradiction, blue = aligned" },
      { file: "contra_04_price_and_model.png",        title: "Price + Model Predictions",
        desc: "Full price history annotated with 14 sources + test-period predictions" },
      { file: "contra_05_sentiment_vs_return_scatter.png", title: "Sentiment vs D+10 Return",
        desc: "Scatter: sentiment score vs D+10 cumulative return per source" },
    ],
    findings: [
      "14 sources analysed: 4 contradictions (Buy rec + Negative outlook)",
      "D+1: market followed Buy recommendation in 100% of dated contradictions",
      "D+10: 75% reverted to negative outlook — short-term contrarian, long-term bearish",
      "Investors look through earnings declines for high-yield OGDC positions",
      "⚠ Sample of 4 — preliminary finding requiring larger dataset validation",
    ],
  },
  {
    id: "enhanced",
    label: "Enhanced ML",
    icon: "⊕",
    color: "#a78bfa",
    charts: [
      { file: "eml_01_clf_comparison.png",      title: "Classification Comparison",
        desc: "ROC-AUC and Accuracy: original vs +trend features for RF and GBM" },
      { file: "eml_02_reg_comparison.png",      title: "Regression Metrics Comparison",
        desc: "MAE, RMSE, R², Directional Accuracy — original vs enhanced" },
      { file: "eml_03_feature_importance.png",  title: "Feature Importance Comparison",
        desc: "Top 20 features: left=original, right=enhanced (teal=new features)" },
      { file: "eml_04_new_feature_value.png",   title: "New Feature Value",
        desc: "BB and volatility features ranked by importance in enhanced model" },
      { file: "eml_05_reg_scatter.png",         title: "Actual vs Predicted",
        desc: "Regression scatter for original (left) and enhanced (right)" },
    ],
    findings: [
      "45 features (enhanced) vs 20 original — marginal AUC improvement",
      "BB_pct_b, BB_sig_breakout_long displaced Month/Quarter from top-20",
      "Economically meaningful: BB position > calendar effects as predictor",
      "R² improvement +0.0001 — original features already near ceiling",
      "GBM classifier slightly degraded: 45-feature space caused CV overfitting",
    ],
  },
];

// ── Stat result cards ─────────────────────────────────────────────────────────
const STAT_RESULTS = [
  { label: "Best Distribution",  value: "Student-t",  sub: "AIC gap = 663 pts vs Normal",   color: C.cyan },
  { label: "ANOVA F-stat",       value: "0.852",      sub: "p = 0.530 — No year difference", color: C.purple },
  { label: "ARMA Order",         value: "(0,3)",      sub: "AIC = 7,390.80",                 color: C.green },
  { label: "GARCH Winner",       value: "GJR",        sub: "Leverage γ = 0.33",              color: C.amber },
  { label: "BB Best Strategy",   value: "Mean Rev",   sub: "Sharpe 0.57 · MaxDD −16.5%",    color: "#38bdf8" },
  { label: "Runs Test",          value: "p = 0.97",   sub: "Returns are random at sign level",color: "#4ade80" },
  { label: "KS Test (Student-t)","value": "p = 0.237","sub": "Only dist to pass KS test",   color: "#f472b6" },
  { label: "Shock Half-Life",    value: "2.8 days",   sub: "GARCH persistence α+β = 0.78",  color: C.amber },
];

// ══════════════════════════════════════════════════════════════════════════════
// COMPONENTS
// ══════════════════════════════════════════════════════════════════════════════

function KpiCard({ label, value, sub, delta, color }) {
  return (
    <div className="kpi-card" style={{
      background: `linear-gradient(135deg, ${C.bg2} 0%, ${C.bg3} 100%)`,
      border: `1px solid ${color}28`,
      borderRadius: 14,
      padding: "18px 22px",
      position: "relative",
      overflow: "hidden",
      cursor: "default",
      "--kpi-shadow": `0 12px 40px ${color}22`,
    }}>
      {/* glow blob */}
      <div style={{
        position: "absolute", top: -20, right: -20,
        width: 80, height: 80, borderRadius: "50%",
        background: `radial-gradient(circle, ${color}22 0%, transparent 70%)`,
        pointerEvents: "none",
      }} />
      <p style={{ margin: 0, fontSize: 11, fontWeight: 700, letterSpacing: "0.1em",
                  color: C.textSecondary, textTransform: "uppercase", fontFamily: "'Space Mono', monospace" }}>
        {label}
      </p>
      <p style={{ margin: "6px 0 2px", fontSize: 28, fontWeight: 800,
                  color: color, fontFamily: "'Space Mono', monospace", lineHeight: 1 }}>
        {value}
      </p>
      {sub && <p style={{ margin: 0, fontSize: 11, color: C.textMuted }}>{sub}</p>}
      {delta && (
        <p style={{ margin: "4px 0 0", fontSize: 12, fontWeight: 600,
                    color: delta > 0 ? C.green : C.red }}>
          {delta > 0 ? "▲" : "▼"} {Math.abs(delta).toFixed(2)}%
        </p>
      )}
    </div>
  );
}

KpiCard.propTypes = {
  label: PropTypes.string.isRequired,
  value: PropTypes.oneOfType([PropTypes.string, PropTypes.number]).isRequired,
  sub:   PropTypes.string,
  delta: PropTypes.number,
  color: PropTypes.string,
};
KpiCard.defaultProps = { sub: null, delta: null, color: C.cyan };

function SectionBadge({ section, active, onClick }) {
  return (
    <button onClick={onClick} className={`section-badge${active ? " active" : ""}`} style={{
      display: "flex", alignItems: "center", gap: 8,
      padding: "10px 16px", borderRadius: 10,
      background: active ? `${section.color}18` : "transparent",
      border: `1px solid ${active ? section.color + "55" : C.border}`,
      color: active ? section.color : C.textSecondary,
      fontSize: 13, fontWeight: 600, cursor: "pointer",
      transition: "all 0.18s", whiteSpace: "nowrap",
      fontFamily: "'Space Mono', monospace",
    }}>
      <span style={{ fontSize: 16 }}>{section.icon}</span>
      {section.label}
    </button>
  );
}

SectionBadge.propTypes = {
  section: PropTypes.shape({
    color: PropTypes.string.isRequired,
    icon:  PropTypes.string.isRequired,
    label: PropTypes.string.isRequired,
  }).isRequired,
  active:  PropTypes.bool.isRequired,
  onClick: PropTypes.func.isRequired,
};

function ChartCard({ chart, sectionColor }) {
  const [expanded, setExpanded] = useState(false);

  const handleKeyDown = (e) => {
    if (e.key === "Enter" || e.key === " ") setExpanded(true);
  };
  const handleCloseKeyDown = (e) => {
    if (e.key === "Enter" || e.key === " " || e.key === "Escape") setExpanded(false);
  };

  return (
    <div className="chart-card" style={{
      background: `linear-gradient(135deg, ${C.bg2} 0%, ${C.bg3} 100%)`,
      border: `1px solid ${C.border}`,
      borderRadius: 14, overflow: "hidden",
      "--chart-shadow": `0 8px 32px ${sectionColor}15`,
      "--chart-border": `${sectionColor}44`,
    }}>
      <img
        src={`/${chart.file}`}
        alt={chart.title}
        role="button"
        tabIndex={0}
        onClick={() => setExpanded(true)}
        onKeyDown={handleKeyDown}
        style={{ width: "100%", display: "block", cursor: "zoom-in",
                 background: "#040810" }}
        onError={e => { e.target.style.display = "none"; }}
      />
      <div style={{ padding: "12px 16px 14px" }}>
        <p style={{ margin: "0 0 4px", fontSize: 13, fontWeight: 700,
                    color: C.textPrimary, fontFamily: "'Space Mono', monospace" }}>
          {chart.title}
        </p>
        <p style={{ margin: 0, fontSize: 11, color: C.textMuted, lineHeight: 1.5 }}>
          {chart.desc}
        </p>
      </div>

      {/* Lightbox */}
      {expanded && (
        <div
          role="dialog"
          aria-modal="true"
          aria-label={`Expanded view of ${chart.title}`}
          tabIndex={0}
          onClick={() => setExpanded(false)}
          onKeyDown={handleCloseKeyDown}
          style={{
            position: "fixed", inset: 0, background: "#000000dd",
            display: "flex", alignItems: "center", justifyContent: "center",
            zIndex: 9999, cursor: "zoom-out", backdropFilter: "blur(4px)",
          }}
        >
          <img src={`/${chart.file}`} alt={chart.title}
               style={{ maxWidth: "92vw", maxHeight: "90vh", borderRadius: 12,
                        boxShadow: `0 0 80px ${sectionColor}33` }} />
          <div style={{ position: "absolute", bottom: 24, left: "50%",
                        transform: "translateX(-50%)", color: C.textSecondary,
                        fontSize: 12, fontFamily: "'Space Mono', monospace" }}>
            Click anywhere to close
          </div>
        </div>
      )}
    </div>
  );
}

ChartCard.propTypes = {
  chart: PropTypes.shape({
    file:  PropTypes.string.isRequired,
    title: PropTypes.string.isRequired,
    desc:  PropTypes.string.isRequired,
  }).isRequired,
  sectionColor: PropTypes.string.isRequired,
};

function FindingsList({ findings, color }) {
  return (
    <div style={{
      background: `linear-gradient(135deg, ${C.bg2} 0%, ${C.bg1} 100%)`,
      border: `1px solid ${color}22`,
      borderLeft: `3px solid ${color}`,
      borderRadius: "0 12px 12px 0",
      padding: "16px 20px",
    }}>
      <p style={{ margin: "0 0 10px", fontSize: 11, fontWeight: 700,
                  color: color, textTransform: "uppercase", letterSpacing: "0.1em",
                  fontFamily: "'Space Mono', monospace" }}>
        Key Findings
      </p>
      {findings.map((f) => (
        <div key={f.slice(0, 40)} style={{ display: "flex", gap: 10, marginBottom: 8 }}>
          <span style={{ color: color, fontSize: 12, marginTop: 2, flexShrink: 0 }}>◆</span>
          <p style={{ margin: 0, fontSize: 12.5, color: C.textSecondary, lineHeight: 1.6 }}>{f}</p>
        </div>
      ))}
    </div>
  );
}

FindingsList.propTypes = {
  findings: PropTypes.arrayOf(PropTypes.string).isRequired,
  color:    PropTypes.string.isRequired,
};

// ── Custom Recharts tooltip ───────────────────────────────────────────────────
function CustomTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  return (
    <div style={{
      background: C.bg3, border: `1px solid ${C.border}`,
      borderRadius: 10, padding: "10px 14px",
      fontFamily: "'Space Mono', monospace", fontSize: 11,
    }}>
      <p style={{ margin: "0 0 6px", color: C.textSecondary }}>{label}</p>
      {payload.map((p) => (
        <p key={p.dataKey || p.name} style={{ margin: "2px 0", color: p.color || C.cyan }}>
          {p.name}: <strong>{typeof p.value === "number" ? p.value.toFixed(2) : p.value}</strong>
        </p>
      ))}
    </div>
  );
}

CustomTooltip.propTypes = {
  active:  PropTypes.bool,
  payload: PropTypes.arrayOf(PropTypes.shape({
    dataKey: PropTypes.string,
    name:    PropTypes.string,
    value:   PropTypes.oneOfType([PropTypes.number, PropTypes.string]),
    color:   PropTypes.string,
  })),
  label: PropTypes.oneOfType([PropTypes.string, PropTypes.number]),
};
CustomTooltip.defaultProps = { active: false, payload: [], label: "" };

// ══════════════════════════════════════════════════════════════════════════════
// PAGES
// ══════════════════════════════════════════════════════════════════════════════

function OverviewPage({ priceData, kpis, annualStats }) {
  const chartData = priceData.slice(-400);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 32 }}>

      {/* KPI grid */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: 16 }}>
        <KpiCard label="Latest Price"   value={`PKR ${kpis.latestPrice}`}  sub="Apr 3, 2026"          color={C.cyan}   />
        <KpiCard label="Total Return"   value={`${kpis.totalReturn > 0 ? "+" : ""}${kpis.totalReturn}%`} sub="Jan 2020 → Apr 2026" color={C.green}  />
        <KpiCard label="Annual Sharpe"  value={kpis.sharpe}                sub="Daily mean / std × √252" color={C.purple} />
        <KpiCard label="Max Drawdown"   value={`${kpis.maxDrawdown}%`}     sub="Peak-to-trough"       color={C.red}    />
        <KpiCard label="Current HV-20"  value={`${kpis.currentHV}%`}       sub="Annualised volatility" color={C.amber}  />
        <KpiCard label="Observations"   value={kpis.observations?.toLocaleString()} sub="Trading days"  color={C.cyan}   />
      </div>

      {/* Price chart */}
      <div style={{
        background: `linear-gradient(135deg, ${C.bg2} 0%, ${C.bg3} 100%)`,
        border: `1px solid ${C.border}`, borderRadius: 16, padding: 24,
      }}>
        <p style={{ margin: "0 0 20px", fontSize: 14, fontWeight: 700,
                    color: C.textPrimary, fontFamily: "'Space Mono', monospace",
                    display: "flex", alignItems: "center", gap: 10 }}>
          <span style={{ color: C.cyan }}>◈</span> OGDC Price + Moving Averages (Last 400 Days)
        </p>
        <ResponsiveContainer width="100%" height={320}>
          <ComposedChart data={chartData} margin={{ top: 4, right: 4, bottom: 4, left: 4 }}>
            <CartesianGrid stroke={C.border} strokeDasharray="3 3" />
            <XAxis dataKey="Date" tick={{ fill: C.textMuted, fontSize: 10 }}
                   tickFormatter={d => d?.slice(2, 7)} interval={30} />
            <YAxis tick={{ fill: C.textMuted, fontSize: 10 }} />
            <Tooltip content={<CustomTooltip />} />
            <Legend wrapperStyle={{ fontSize: 11, color: C.textSecondary }} />
            <Area dataKey="price" name="Price" stroke={C.cyan} fill={`${C.cyan}12`}
                  strokeWidth={1.8} dot={false} />
            <Line dataKey="SMA20"  name="SMA-20"  stroke={C.amber}  strokeWidth={1.2} dot={false} strokeDasharray="4 2" />
            <Line dataKey="SMA50"  name="SMA-50"  stroke={C.green}  strokeWidth={1.4} dot={false} strokeDasharray="6 2" />
            <Line dataKey="SMA200" name="SMA-200" stroke={C.red}    strokeWidth={1.6} dot={false} />
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      {/* Returns bar + HV chart */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20 }}>
        <div style={{
          background: `linear-gradient(135deg, ${C.bg2} 0%, ${C.bg3} 100%)`,
          border: `1px solid ${C.border}`, borderRadius: 16, padding: 24,
        }}>
          <p style={{ margin: "0 0 16px", fontSize: 13, fontWeight: 700,
                      color: C.textPrimary, fontFamily: "'Space Mono', monospace" }}>
            <span style={{ color: C.purple }}>∑</span> Mean Return by Year
          </p>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={annualStats} margin={{ top: 4, right: 4, bottom: 4, left: 4 }}>
              <CartesianGrid stroke={C.border} strokeDasharray="3 3" />
              <XAxis dataKey="Year" tick={{ fill: C.textMuted, fontSize: 11 }} />
              <YAxis tick={{ fill: C.textMuted, fontSize: 10 }}
                     tickFormatter={v => `${v.toFixed(2)}%`} />
              <Tooltip content={<CustomTooltip />} />
              <ReferenceLine y={0} stroke={C.textMuted} strokeOpacity={0.5} />
              <Bar dataKey="mean" name="Mean Return %" radius={[4, 4, 0, 0]}
                   isAnimationActive={true}>
                {annualStats.map((entry) => (
                  <Cell key={entry.Year} fill={entry.mean >= 0 ? C.cyan : C.red} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div style={{
          background: `linear-gradient(135deg, ${C.bg2} 0%, ${C.bg3} 100%)`,
          border: `1px solid ${C.border}`, borderRadius: 16, padding: 24,
        }}>
          <p style={{ margin: "0 0 16px", fontSize: 13, fontWeight: 700,
                      color: C.textPrimary, fontFamily: "'Space Mono', monospace" }}>
            <span style={{ color: C.amber }}>〜</span> 20-Day Historical Volatility
          </p>
          <ResponsiveContainer width="100%" height={220}>
            <AreaChart data={chartData} margin={{ top: 4, right: 4, bottom: 4, left: 4 }}>
              <CartesianGrid stroke={C.border} strokeDasharray="3 3" />
              <XAxis dataKey="date" tick={{ fill: C.textMuted, fontSize: 10 }}
                     tickFormatter={d => d?.slice(2, 7)} interval={30} />
              <YAxis tick={{ fill: C.textMuted, fontSize: 10 }}
                     tickFormatter={v => `${v.toFixed(0)}%`} />
              <Tooltip content={<CustomTooltip />} />
              <Area dataKey="hv20" name="HV-20 Ann.%" stroke={C.amber}
                    fill={`${C.amber}18`} strokeWidth={1.5} dot={false} />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Stat results grid */}
      <div>
        <p style={{ margin: "0 0 14px", fontSize: 13, fontWeight: 700,
                    color: C.textPrimary, fontFamily: "'Space Mono', monospace" }}>
          <span style={{ color: C.cyan }}>◆</span> All Statistical Results
        </p>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: 12 }}>
          {STAT_RESULTS.map((s) => (
            <div key={s.label} style={{
              background: C.bg2, border: `1px solid ${s.color}22`,
              borderLeft: `3px solid ${s.color}`,
              borderRadius: "0 10px 10px 0", padding: "12px 16px",
            }}>
              <p style={{ margin: "0 0 2px", fontSize: 10, color: C.textMuted,
                          textTransform: "uppercase", letterSpacing: "0.08em",
                          fontFamily: "'Space Mono', monospace" }}>{s.label}</p>
              <p style={{ margin: "0 0 2px", fontSize: 20, fontWeight: 800,
                          color: s.color, fontFamily: "'Space Mono', monospace" }}>{s.value}</p>
              <p style={{ margin: 0, fontSize: 11, color: C.textMuted }}>{s.sub}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

OverviewPage.propTypes = {
  priceData:   PropTypes.arrayOf(PropTypes.object).isRequired,
  kpis:        PropTypes.object.isRequired,
  annualStats: PropTypes.arrayOf(PropTypes.object).isRequired,
};

function SectionPage({ section }) {
  const cols = section.charts.length === 1 ? 1
    : section.charts.length <= 3 ? section.charts.length
    : 2;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 24 }}>
      <FindingsList findings={section.findings} color={section.color} />
      <div style={{
        display: "grid",
        gridTemplateColumns: `repeat(${cols}, 1fr)`,
        gap: 20,
      }}>
        {section.charts.map((chart) => (
          <ChartCard key={chart.file} chart={chart} sectionColor={section.color} />
        ))}
      </div>
    </div>
  );
}

SectionPage.propTypes = {
  section: PropTypes.shape({
    charts:   PropTypes.arrayOf(PropTypes.object).isRequired,
    findings: PropTypes.arrayOf(PropTypes.string).isRequired,
    color:    PropTypes.string.isRequired,
  }).isRequired,
};

// ══════════════════════════════════════════════════════════════════════════════
// APP
// ══════════════════════════════════════════════════════════════════════════════
export default function App() {
  const [activePage, setActivePage] = useState("overview");
  const [priceData, setPriceData]   = useState([]);
  const [kpis, setKpis]             = useState({});
  const [annualStats, setAnnualStats] = useState([]);
  const [loading, setLoading]       = useState(true);
  const scrollRef = useRef(null);

  useEffect(() => {
    Promise.all([
      fetch(PRICE_DATA_URL).then(r => r.json()).catch(() => []),
      fetch("/kpis.json").then(r => r.json()).catch(() => ({})),
      fetch("/annual_stats.json").then(r => r.json()).catch(() => []),
    ]).then(([price, kpi, annual]) => {
      setPriceData(price);
      setKpis(kpi);
      setAnnualStats(annual);
      setLoading(false);
    });
  }, []);

  // Scroll to top on page change
  useEffect(() => {
    if (scrollRef.current) scrollRef.current.scrollTop = 0;
  }, [activePage]);

  const activeSection = SECTIONS.find(s => s.id === activePage);

  return (
    <div style={{
      minHeight: "100vh",
      background: `radial-gradient(ellipse at 20% 0%, #0a1f3380 0%, transparent 50%),
                   radial-gradient(ellipse at 80% 100%, #00e5ff08 0%, transparent 50%),
                   ${C.bg0}`,
      color: C.textPrimary,
      fontFamily: "'Space Mono', 'Courier New', monospace",
      display: "flex", flexDirection: "column",
    }}>

      {/* ── Top nav ─────────────────────────────────────────────────────────── */}
      <nav style={{
        position: "sticky", top: 0, zIndex: 100,
        background: `${C.bg0}ee`, backdropFilter: "blur(16px)",
        borderBottom: `1px solid ${C.border}`,
        padding: "0 24px", height: 60,
        display: "flex", alignItems: "center", justifyContent: "space-between",
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
          <div style={{
            width: 34, height: 34, borderRadius: 8,
            background: `linear-gradient(135deg, ${C.cyan}33, ${C.cyan}11)`,
            border: `1px solid ${C.cyan}44`,
            display: "flex", alignItems: "center", justifyContent: "center",
            fontSize: 16,
          }}>📈</div>
          <div>
            <p style={{ margin: 0, fontSize: 13, fontWeight: 800, color: C.cyan,
                        letterSpacing: "0.05em" }}>OGDC ANALYTICS</p>
            <p style={{ margin: 0, fontSize: 9, color: C.textMuted,
                        letterSpacing: "0.12em", textTransform: "uppercase" }}>
              TREND ANALYSIS · PSX
            </p>
          </div>
        </div>

        {/* Desktop nav tabs */}
        <div style={{ display: "flex", alignItems: "center", gap: 4, flexWrap: "wrap" }}>
          <button
            onClick={() => setActivePage("overview")}
            style={{
              padding: "7px 14px", borderRadius: 8, fontSize: 11, fontWeight: 700,
              cursor: "pointer", border: "1px solid",
              fontFamily: "'Space Mono', monospace", letterSpacing: "0.04em",
              background: activePage === "overview" ? `${C.cyan}18` : "transparent",
              borderColor: activePage === "overview" ? `${C.cyan}55` : C.border,
              color: activePage === "overview" ? C.cyan : C.textSecondary,
              transition: "all 0.15s",
            }}>
            ⬡ OVERVIEW
          </button>
          {SECTIONS.map(s => (
            <button key={s.id}
              onClick={() => setActivePage(s.id)}
              style={{
                padding: "7px 12px", borderRadius: 8, fontSize: 10, fontWeight: 700,
                cursor: "pointer", border: "1px solid",
                fontFamily: "'Space Mono', monospace",
                background: activePage === s.id ? `${s.color}18` : "transparent",
                borderColor: activePage === s.id ? `${s.color}55` : C.border,
                color: activePage === s.id ? s.color : C.textSecondary,
                transition: "all 0.15s",
              }}>
              {s.icon} {s.label.toUpperCase().slice(0, 8)}
            </button>
          ))}
        </div>

        <div style={{ fontSize: 10, color: C.textMuted, textAlign: "right" }}>
          <p style={{ margin: 0 }}>Jan 2020 → Apr 2026</p>
          <p style={{ margin: 0 }}>1,552 trading days</p>
        </div>
      </nav>

      {/* ── Main content ─────────────────────────────────────────────────────── */}
      <main ref={scrollRef} style={{ flex: 1, padding: "32px 28px 60px", maxWidth: 1400, margin: "0 auto", width: "100%" }}>

        {loading ? (
          <div style={{ display: "flex", alignItems: "center", justifyContent: "center",
                        height: 400, flexDirection: "column", gap: 20 }}>
            <div style={{ width: 48, height: 48, borderRadius: "50%",
                          border: `3px solid ${C.border}`, borderTop: `3px solid ${C.cyan}`,
                          animation: "spin 1s linear infinite" }} />
            <p style={{ color: C.textMuted, fontSize: 13 }}>Loading market data...</p>
          </div>
        ) : (
          <>
            {/* Page header */}
            <div style={{ marginBottom: 28 }}>
              <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 8 }}>
                <span style={{ fontSize: 22, color: activeSection?.color || C.cyan }}>
                  {activeSection?.icon || "⬡"}
                </span>
                <h1 style={{
                  margin: 0, fontSize: 24, fontWeight: 800, letterSpacing: "0.02em",
                  color: activeSection?.color || C.cyan,
                  fontFamily: "'Space Mono', monospace",
                }}>
                  {activePage === "overview"
                    ? "PROJECT OVERVIEW"
                    : activeSection?.label.toUpperCase()}
                </h1>
              </div>
              <div style={{ height: 2, background: `linear-gradient(90deg, ${activeSection?.color || C.cyan}66, transparent)`, borderRadius: 1 }} />
            </div>

            {activePage === "overview"
              ? <OverviewPage priceData={priceData} kpis={kpis} annualStats={annualStats} />
              : activeSection && <SectionPage section={activeSection} />
            }
          </>
        )}
      </main>

      {/* ── Footer ───────────────────────────────────────────────────────────── */}
      <footer style={{
        borderTop: `1px solid ${C.border}`, padding: "16px 28px",
        display: "flex", justifyContent: "space-between", alignItems: "center",
        background: `${C.bg1}cc`,
      }}>
        <p style={{ margin: 0, fontSize: 10, color: C.textMuted }}>
          OGDC Analytics Platform · Stock Trend Analysis
        </p>
        <p style={{ margin: 0, fontSize: 10, color: C.textMuted }}>
          React · Recharts · Tailored for PSX Equity Research
        </p>
      </footer>

      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&display=swap');
        * { box-sizing: border-box; }
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: ${C.bg0}; }
        ::-webkit-scrollbar-thumb { background: ${C.border}; border-radius: 3px; }
        ::-webkit-scrollbar-thumb:hover { background: ${C.cyan}44; }
        @keyframes spin { to { transform: rotate(360deg); } }
        body { margin: 0; }
        .kpi-card {
          transition: transform 0.2s, box-shadow 0.2s;
        }
        .kpi-card:hover {
          transform: translateY(-3px);
          box-shadow: var(--kpi-shadow);
        }
        .chart-card {
          transition: border-color 0.2s, box-shadow 0.2s;
        }
        .chart-card:hover {
          border-color: var(--chart-border) !important;
          box-shadow: var(--chart-shadow);
        }
      `}</style>
    </div>
  );
}
