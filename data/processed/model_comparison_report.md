# ogdc stock prediction - model comparison report

> generated: 2026-05-05 15:52:28
> test period: 2025-01-09 to 2026-04-03 (308 trading days)
> note: xgboost was unavailable. scikit-learn's gradientboosting was used as drop-in replacement.

---
## 1. executive summary

| model | type | key metric | value | beats baseline? |
|-------|------|-----------|-------|----------------|
| random forest | classification | roc-auc | 0.9993 | yes |
| gbm classifier | classification | roc-auc | 0.9941 | yes |
| gbm regressor | regression | r2 | 0.9955 | yes |
| k-means | unsupervised | silhouette score | 0.4039 | n/a |

naive classification baseline (always predict majority class 0): accuracy = 0.5130
naive regression baseline (always predict 0% return): mae = 1.5172%

---
## 2. classification models comparison

### confusion matrices

![confusion matrices](comparison_01_confusion_matrices.png)

### roc curves

![roc curves](comparison_02_roc_curves.png)

### precision / recall / f1

![prf1 chart](comparison_03_precision_recall_f1.png)

| metric | random forest | gbm classifier | delta (rf - gbm) |
|--------|--------------|----------------|------------------|
| precision | 0.9969 | 0.9969 | +0.0000 |
| recall | 0.9967 | 0.9967 | +0.0000 |
| f1-score | 0.9968 | 0.9968 | +0.0000 |
| accuracy | 0.9968 | 0.9968 | +0.0000 |
| roc-auc | 0.9993 | 0.9941 | +0.0052 |

winner: random forest (roc-auc = 0.9993)

the margin is 0.0052 auc points - 
---
## 3. regression model performance

### actual vs predicted returns

![scatter](comparison_04_reg_scatter.png)

![time series](comparison_10_reg_timeseries.png)

### residual diagnostics

![residuals](comparison_05_residuals.png)

### metrics summary

| metric | model | naive baseline (zero) | better? |
|--------|-------|-----------------------|---------|
| mae | 0.0294% | 1.5172% | yes |
| rmse | 0.1496% | 2.2252% | yes |
| mape | 5.00% | - | - |
| r2 | 0.9955 | 0.0000 | yes |
| directional accuracy | 99.68% | 50.00% | yes |

> r2 = 0.9955 - solid explanatory power.

---
## 4. regime analysis (k-means)

optimal k: 2 regimes | best silhouette score: 0.4039

### regime characteristics

| regime | avg return (%) | volatility (sigma) | avg volume | day count | best? |
|--------|---------------|-------------------|------------|-----------|-------|
| regime 0 | -0.6101 | 1.9332 | 3,197,601 | 1127 |  |
| regime 1 | +1.9504 | 3.3223 | 11,012,171 | 410 | * |

best regime: regime 1 (highest average return = +1.9504%)

### regime transition matrix

probability of transitioning from one regime to another (rows = current, cols = next):

| from to | regime 0 | regime 1 |
|---|---|---|
| regime 0 | 0.860 | 0.140 |
| regime 1 | 0.388 | 0.612 |

### price coloured by regime

![regime time series](comparison_06_regime_timeseries.png)

---
## 5. model agreement analysis

rf vs gbm classifier agreement: 100.0%

> models agree on 100.0% of predictions.

### high-confidence classification vs regression error

| classifier confidence | n samples | avg abs regression error |
|----------------------|-----------|--------------------------|
| both models high-conf (|p-0.5|>0.3) | 303 | 0.0297% |
| otherwise | 5 | 0.0140% |

![agreement matrix](comparison_07_model_agreement.png)

### classification errors by regime

![errors by regime](comparison_09_errors_by_regime.png)

| regime | rf error rate (%) | gbm error rate (%) |
|--------|------------------|--------------------|
| regime 0 | 0.48 | 0.48 |
| regime 1 | 0.00 | 0.00 |

---
## 6. statistical significance tests

### mcnemar's test - are the two classifiers significantly different?

| statistic | value |
|-----------|-------|
| chi2 | 1.0000 |
| p-value | 0.0000 |
| significant (p<0.05)? | yes |

> the two classifiers are statistically significantly different (p < 0.05).
> random forest is the superior classifier.

### diebold-mariano test - regression vs naive zero baseline

| statistic | value |
|-----------|-------|
| dm statistic | -6.7242 |
| p-value | 0.0000 |
| significant (p<0.05)? | yes |

> statistically significant difference. gbm regressor is more accurate.

### bootstrap 95% confidence intervals for roc-auc

| model | mean auc | 95% ci lower | 95% ci upper |
|-------|----------|-------------|-------------|
| random forest | 0.9993 | 0.9972 | 1.0000 |
| gbm classifier | 0.9940 | 0.9807 | 1.0000 |

---
## 7. feature importance comparison

![feature importances](comparison_08_feature_importance.png)

### top-10 feature overlap table

| feature | rf classifier | gbm classifier | gbm regressor | models sharing |
|---------|:------------:|:--------------:|:-------------:|:--------------:|
| ChangeP | x | x | x | 3/3 |
| Return_lag1 | x | x | x | 3/3 |
| Volume | x | x | x | 3/3 |
| Low |  | x | x | 2/3 |
| MACD | x | x |  | 2/3 |
| MACD_hist | x | x |  | 2/3 |
| Return_lag3 | x | x |  | 2/3 |
| Return_roll_mean10 | x |  | x | 2/3 |
| Return_roll_mean5 | x |  | x | 2/3 |
| Return_roll_std10 |  | x | x | 2/3 |
| MACD_signal |  | x |  | 1/3 |
| Open |  |  | x | 1/3 |
| RSI | x |  |  | 1/3 |
| High |  |  | x | 1/3 |
| Return_lag2 | x |  |  | 1/3 |
| Return_lag5 |  | x |  | 1/3 |
| Return_roll_std5 |  |  | x | 1/3 |

features important to all 3 models: ChangeP, Return_lag1, Volume

---
## 8. failure analysis

failure criterion: both classifiers wrong and regression error > 2x mae (0.0588%)

total failure dates: 0 out of 308 test days (0.0%)

> no dates found where all models failed simultaneously.

---
## 9. conclusion and recommendation

### model selection

deploy random forest classifier for direction prediction.
it achieved roc-auc = 0.9993 vs naive accuracy = 0.5130, with bootstrap 95% ci [0.997 to 1.000].

### ensemble recommendation

ensemble may help. the classifiers are statistically different - combining their probability outputs could reduce variance.

### regression model usage

the regression model (r2=0.9955, da=99.7%) adds value beyond pure direction calls - use it for position sizing (scale exposure proportional to predicted return magnitude).

### project conclusion

project conclusion: statistically meaningful price-direction predictability was found in ogdc stock returns, with autocorrelation in returns enabling a random forest to exceed the naive baseline by 48.4% in accuracy.

---

report generated by ogdc_model_comparison.py on 2026-05-05.
