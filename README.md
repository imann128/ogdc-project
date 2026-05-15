# OGDC Stock Price Analysis

Quantitative analysis of Oil and Gas Development Company Limited (OGDC) on the Pakistan Stock Exchange (PSX), using 1,552 daily trading observations from January 2020 to April 2026. The project covers statistical inference, machine learning, GARCH volatility modelling, Bollinger Band backtesting, and sentiment analysis — structured as a reproducible pipeline from raw CSV to interactive dashboard.

---

## Data Source

Daily OHLCV data was obtained from [Investing.com](https://www.investing.com/equities/oil-gas-dev-historical-data) as a manual CSV export. The raw file contains seven columns: Date (MM/DD/YYYY), Price, Open, High, Low, Vol. (abbreviated string format e.g. "4.51M"), and Change % (string with % symbol). The file is stored unmodified in `data/raw/` and never overwritten by any script.

**Exchange:** Pakistan Stock Exchange (PSX)  
**Ticker:** OGDC  
**Period:** 1 January 2020 – 3 April 2026  
**Observations:** 1,552 trading days  

Sentiment data was manually collected and scored from 14 sources:

| # | Source | URL |
|---|---|---|
| 1 | KSEStocks Blog | https://ksestocks.com/blog/ogdc-the-peoples-choice-for-all-time-stock/ |
| 2 | StockAnalysis.com | https://stockanalysis.com/quote/psx/OGDC/ |
| 3 | Investing.com Technical | https://www.investing.com/equities/oil---gas-dev-technical |
| 4 | TradingView Community Ideas | https://www.tradingview.com/symbols/PSX-OGDC/ideas/ |
| 5 | YouTube Video | https://youtu.be/_NOSb2kgWKQ |
| 6 | DefencePK Forum | https://defencepk.com/forums/threads/...17551/ |
| 7 | Quora — Blue-Chip List | https://www.quora.com/Which-are-the-blue-chip-companies-in-Pakistan |
| 8 | TradersUnion Forecast | https://tradersunion.com/currencies/forecast/ogdc-pkr/daily-and-weekly/ |
| 9 | Mettis Global — Q4FY24 Profit Collapse | https://mettisglobal.news/ogdcs-profit-collapse-sinks-oil-gas-sector-in-q4fy24/ |
| 10 | Dawn.com — OGDCL Profit Slumps 33% | https://www.dawn.com/news/1895049 |
| 11 | Mettis Global — FY25 Profit Falls | https://mettisglobal.news/OGDCs-FY25-profit-falls-to-Rs170bn-shareholders-to-receive-record-dividend-55420 |
| 12 | ProPakistani — FY24 Profit Falls 7% | https://propakistani.pk/2024/09/23/ogdc-profit-falls-7-to-rs-208-9-billion-in-fy24/ |
| 13 | Insight Research / Topline Securities | Insight Research on OGDC Growth Potential (June 2024) |
| 14 | IGI Securities Strategy Report | IGI Securities Strategy 2024 Report (December 2023) |

---

## Key Findings

**Statistical Tests**

- Daily returns follow a **Student-t distribution with ν ≈ 2.8 degrees of freedom**, not the Normal distribution (AIC gap of 663 points). At this tail weight, a daily move exceeding ±5% is approximately 14 times more likely than a Gaussian model would predict — a direct consequence of PSX circuit-breaker events and thin liquidity.
- **ANOVA finds no significant difference in mean returns across calendar years** (F = 0.852, p = 0.530, η² = 0.003). Despite price rising from PKR 69.55 to PKR 333.90, day-to-day arithmetic returns are statistically indistinguishable across years. Kruskal-Wallis confirms this nonparametrically (H = 3.82, p = 0.701).
- **ARMA(0,3) is the optimal time series model** (AIC = 7,390.80). Significant ACF spikes at lags 1, 3, 12, 22, and 29 indicate short-horizon autocorrelation, but Ljung-Box confirms residuals are white noise — meaning the model captures all detectable structure. The 30-day forecast degenerates to the unconditional mean within a few periods.
- **The Runs Test fails to reject randomness** (Z = 0.043, p = 0.966). Knowing whether today's return is positive or negative provides no information about tomorrow's direction — consistent with weak-form market efficiency at the sign level.
- **Spearman correlation between HV-20 and daily returns is ρ = 0.013** (p = 0.609). Volatility predicts the magnitude of moves, not their direction.

**Machine Learning**

- When the `ChangeP` column (same-day percentage change from the raw CSV) is included as a feature, classifiers achieve ~99.7% accuracy and AUC > 0.999. This is data leakage — `ChangeP` is numerically equivalent to the classification target. `ogdc_enhanced_ml.py` evaluates all models with this column excluded, dropping accuracy to **53–54%**, which is the honest figure. This modest edge above the 51.3% majority-class baseline is consistent with the short-horizon autocorrelation confirmed by the Ljung-Box test.
- **K-Means identifies two distinct market regimes** (Silhouette = 0.40): a bearish regime covering 1,127 days (avg return −0.61%/day) and a bullish regime covering 410 days (avg return +1.95%/day). The transition matrix shows 61% probability of remaining in the bullish regime once entered.

**GARCH Volatility**

- **GJR-GARCH(1,1) is the best-fitting model** (AIC = 7,040). The leverage parameter γ = 0.33 confirms that negative shocks amplify future volatility 33% more than positive shocks of equal size — consistent with PSX investor behaviour reacting more strongly to earnings misses than to beats.
- GARCH(1,1) persistence α+β = 0.78 implies a **shock half-life of 2.8 trading days** — considerably faster mean-reversion than developed market equities where half-lives of 20–30 days are typical.

**Backtesting**

- **BB Mean Reversion** matches Buy-and-Hold's Sharpe ratio (0.57) while reducing maximum drawdown from −41.2% to −16.5%, making it the most practically useful active strategy. Out-of-sample (2024–2026), BB Breakout showed the strongest performance (Sharpe 1.04, win rate 66.7%), suggesting a regime shift toward momentum-driven price action after 2023.

**Sentiment**

- Four of the 14 sources contained contradictions — Buy recommendations paired with negative fundamental outlooks. On both dated contradiction events, the stock moved upward the following day (+1.8% and +2.1%). Both showed negative D+10 cumulative returns, suggesting a short-term contrarian reaction followed by reversion toward the fundamental outlook.

---

## Project Structure

```
ogdc-project/
│
├── data/
│   ├── raw/                          original files, never modified by any script
│   │   ├── Oil_and_Gas_Development_Co_Stock_Price_History.csv
│   │   ├── USD_PKR_Historical_Data.csv
│   │   └── ogdc_sentiment_results.csv
│   │
│   └── processed/                    all script outputs land here
│       ├── ogdc_cleaned.csv
│       ├── ogdc_with_regimes.csv
│       ├── ogdc_trend_features.csv
│       ├── ogdc_garch_results.csv
│       ├── ogdc_backtest_results.csv
│       ├── model_predictions.csv
│       ├── stat222_results.csv
│       ├── contradiction_analysis_full.csv
│       └── contradiction_summary.csv
│
├── scripts/
│   ├── ogdc_loader_cleaner.py           Step 1
│   ├── ogdc_analysis.py                 Step 2
│   ├── ogdc_stat222.py                  Step 3
│   ├── ogdc_trend_analysis.py           Step 4
│   ├── ogdc_garch.py                    Step 5
│   ├── ogdc_backtest.py                 Step 6
│   ├── ogdc_model_comparison.py         Step 7
│   ├── ogdc_enhanced_ml.py              Step 8
│   ├── ogdc_contradiction_analysis.py   Step 9
│   ├── export_frontend_data.py          Step 10
│   ├── ogdc_dashboard.py                Streamlit dashboard
│   └── build_report.js                  Word report generator (Node.js)
│
├── outputs/
│   ├── images/       
|                   all PNG charts saved here automatically
|
│
├── frontend/
│   ├── public/                          JSONs and PNGs go here (via export_frontend_data.py)
│   ├── src/
│   │   ├── App.jsx
│   │   └── main.jsx
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js
│   └── vercel.json
│
└── README.md
```

---

## Dependencies

The project uses only standard scientific Python. No XGBoost, no statsmodels, no arch library — scikit-learn's GradientBoosting replaces XGBoost, regression diagnostics are computed with numpy directly, and all GARCH models are implemented from scratch using scipy.optimize.

```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn
```

For the Streamlit dashboard:

```bash
pip install streamlit plotly pillow
```

For the React frontend:

```bash
cd frontend
npm install
```

For the Word report generator:

```bash
npm install -g docx
```

---

## Running the Pipeline

Each script reads from `data/processed/` and writes back to `data/processed/` or `outputs/images/`. Run in order — later scripts depend on files produced by earlier ones.

### Using the master runner (recommended)

Run everything from the project root:

```bash
python run_pipeline.py
```

Run a single step by number:

```bash
python run_pipeline.py --step 2
```

Run multiple specific steps:

```bash
python run_pipeline.py --step 1 4 5
```

Steps have dependencies. If you skip a step, any later step that relies on its output will fail. Always start from the earliest step in the dependency chain your target script needs.

| Step | Script | Description |
|------|--------|-------------|
| 1 | `ogdc_loader_cleaner.py` | Load & clean raw CSV |
| 2 | `ogdc_analysis.py` | Feature engineering + ML models |
| 3 | `ogdc_stat222.py` | STAT-222 statistical tests |
| 4 | `ogdc_trend_analysis.py` | Moving averages, BB, volatility |
| 5 | `ogdc_garch.py` | GARCH volatility modelling |
| 6 | `ogdc_backtest.py` | Bollinger Band backtesting |
| 7 | `ogdc_model_comparison.py` | Cross-model comparison + markdown report |
| 8 | `ogdc_enhanced_ml.py` | Enhanced ML with trend features |
| 9 | `ogdc_contradiction_analysis.py` | Sentiment contradiction analysis |
| 10 | `export_frontend_data.py` | Export JSON data for React frontend |

### Running scripts individually

```bash
python scripts/ogdc_loader_cleaner.py
python scripts/ogdc_analysis.py
python scripts/ogdc_stat222.py
python scripts/ogdc_trend_analysis.py
python scripts/ogdc_garch.py
python scripts/ogdc_backtest.py
python scripts/ogdc_model_comparison.py
python scripts/ogdc_enhanced_ml.py
python scripts/ogdc_contradiction_analysis.py
python scripts/export_frontend_data.py
```

All scripts resolve paths relative to the project root automatically. You do not need to set any environment variables or path configuration.

---

## Script Reference

### `ogdc_loader_cleaner.py`

Cleans the raw Investing.com CSV. Handles: UTF-8 BOM stripping, column name normalisation for Windows encoding variants, row-by-row date parsing from MM/DD/YYYY with per-row error handling, ascending sort (Investing.com exports newest-first), volume string conversion from abbreviated format (4.51M → 4,510,000), percentage symbol removal. Computes `Return` (daily % change) and drops the first row which has no prior day.

**Input:** `data/raw/Oil_and_Gas_Development_Co_Stock_Price_History.csv`  
**Output:** `data/processed/ogdc_cleaned.csv`

---

### `ogdc_analysis.py`

Feature engineering and four ML models on a **70/15/15 chronological split** — train, validation (hyperparameter tuning only), test (evaluated once at the end).

Features: lag returns at 1, 2, 3, 5 days; 5-day and 10-day rolling mean and standard deviation; RSI-14 via Wilder's EMA; MACD (12, 26, 9) and MACD histogram; day-of-week, month, quarter. NaN rows from rolling warmup are dropped, not filled.

Models: Random Forest classifier, GBM classifier (GridSearchCV + TimeSeriesSplit), GBM regressor, K-Means regime detection. Saves full predictions (validation and test rows, actual vs predicted) to `model_predictions.csv`.

**Input:** `data/processed/ogdc_cleaned.csv`  
**Output:** `data/processed/ogdc_with_regimes.csv`, `data/processed/model_predictions.csv`, `outputs/images/ogdc_*.png`

---

### `ogdc_stat222.py`

All formal statistical tests. ADF stationarity on price and returns. Jarque-Bera and Shapiro-Wilk normality. MLE distribution fitting across five candidates (Normal, Student-t, Laplace, Logistic, Cauchy) with AIC/BIC/KS comparison. One-way and two-way ANOVA with Levene's assumption check and Bonferroni-corrected post-hoc pairwise tests. OLS multiple regression with manual VIF and Breusch-Pagan diagnostics. ARMA grid search with Ljung-Box residual validation and 30-day forecast. Kruskal-Wallis, Mann-Whitney U, Spearman rank correlation, and Runs Test.

**Input:** `data/processed/ogdc_cleaned.csv`  
**Output:** `data/processed/stat222_results.csv`, `outputs/images/stat_*.png`

---

### `ogdc_trend_analysis.py`

Technical indicator computation across the full series. No train/test split — all indicators are descriptive. Computes: SMA 20/50/200, EMA 12/26, WMA 20 with Golden/Death Cross detection; Bollinger Bands (20-day 2σ), %B, bandwidth, squeeze detection, and four signal types; HV-20/60/120, Parkinson, and Garman-Klass volatility estimators; additive time series decomposition (252-day centred trend, day-of-year seasonal, residual); ACF and PACF to lag 40 using manual dot-product and Levinson-Durbin implementations.

**Input:** `data/processed/ogdc_cleaned.csv`  
**Output:** `data/processed/ogdc_trend_features.csv`, `outputs/images/trend_*.png`

---

### `ogdc_garch.py`

Four GARCH family models implemented from scratch using `scipy.optimize` for MLE — no external arch library. ARCH(1), GARCH(1,1), EGARCH(1,1), GJR-GARCH(1,1). Reports log-likelihood, AIC, BIC, persistence, and half-life for each. Generates news impact curves, 30-day volatility forecast with 95% confidence band, and standardised residual diagnostics.

**Input:** `data/processed/ogdc_trend_features.csv`  
**Output:** `data/processed/ogdc_garch_results.csv`, `outputs/images/garch_*.png`

---

### `ogdc_backtest.py`

Four Bollinger Band strategies backtested against Buy-and-Hold and SMA-50 crossover. Transaction cost: 0.17% round-trip. Stop-loss: 3% hard stop. Walk-forward split: 70/30. Strategies: BB Mean Reversion, BB Breakout, Squeeze Breakout, %B Momentum. Reports total return, Sharpe ratio, maximum drawdown, win rate, and trade count per strategy. Includes rolling 12-month Sharpe panel.

**Input:** `data/processed/ogdc_trend_features.csv`  
**Output:** `data/processed/ogdc_backtest_results.csv`, `outputs/images/bt_*.png`

---

### `ogdc_model_comparison.py`

Retrains all ML models and generates a cross-model report. Side-by-side confusion matrices, overlaid ROC curves with 95% bootstrap confidence intervals, McNemar's test between classifiers, Diebold-Mariano test for the regressor against a naive zero-return baseline, feature importance comparison across all three supervised models, and a failure analysis identifying dates where all models failed simultaneously.

**Input:** `data/processed/ogdc_with_regimes.csv`  
**Output:** `outputs/reports/model_comparison_report.md`, `outputs/images/comparison_*.png`

---

### `ogdc_enhanced_ml.py`

Compares model performance across two feature sets: the original 20 features and an enhanced 45-feature set that adds Bollinger Band, moving average, and volatility indicators from the trend analysis step. This is also where **performance without the `ChangeP` column is formally evaluated** — both feature sets are tested with `ChangeP` excluded to produce the realistic deployment accuracy figures.

**Input:** `data/processed/ogdc_with_regimes.csv`, `data/processed/ogdc_trend_features.csv`  
**Output:** `data/processed/ogdc_enhanced_results.csv`, `outputs/images/eml_*.png`

---

### `ogdc_contradiction_analysis.py`

Sentiment contradiction analysis on 14 manually scored sources. Identifies cases where a Buy recommendation conflicts with a negative fundamental outlook and maps each source's publication date to subsequent price windows (D+1, D+3, D+5, D+10) from the cleaned price series. Computes follow-recommendation rate, follow-sentiment rate, model directional accuracy, and Diebold-Mariano test on contradiction days.

**Input:** `data/raw/ogdc_sentiment_results.csv`, `data/processed/model_predictions.csv`  
**Output:** `data/processed/contradiction_analysis_full.csv`, `data/processed/contradiction_summary.csv`, `outputs/images/contra_*.png`

---

### `export_frontend_data.py`

Generates the three JSON files the React dashboard reads — `price_data.json` (full OHLCV series with computed indicators), `kpis.json` (headline metrics), `annual_stats.json` (year-by-year return statistics) — and copies all PNG charts from `outputs/images/` to `frontend/public/`. Run this last, after all analysis scripts have completed.

**Output:** `frontend/public/price_data.json`, `frontend/public/kpis.json`, `frontend/public/annual_stats.json`, all PNGs

---

## Dashboards

### Streamlit

```bash
streamlit run scripts/ogdc_dashboard.py
```

Opens at `http://localhost:8501`. Nine-tab layout covering EDA, statistical tests, ML models, GARCH, trend analysis, backtesting, sentiment, and the enhanced ML comparison.

### React

Run `export_frontend_data.py` first to populate `frontend/public/`. Then:

```bash
cd frontend
npm run dev
```

Opens at `http://localhost:5173`. For a production build:

```bash
npm run build
```

Output is in `frontend/dist/`.

---

## Limitations

**Data leakage:** The `ChangeP` column in the raw Investing.com CSV is numerically equivalent to the daily return — including it as a feature makes the classification target trivially predictable (~99.7% accuracy). `ogdc_enhanced_ml.py` evaluates models with this column excluded. The honest accuracy without it is 53–54%.

**EGARCH convergence:** The EGARCH(1,1) implementation uses Nelder-Mead which can be sensitive to initialisation. In some runs it fails to converge. GJR-GARCH is the recommended model for this dataset.

**Sentiment sample size:** Fourteen sources and four contradiction events are insufficient for statistically robust conclusions. The contradiction analysis identifies a pattern, not a finding.

**Manual data download:** Investing.com does not provide a free programmatic API. The raw CSV must be re-downloaded manually to extend the date range.
