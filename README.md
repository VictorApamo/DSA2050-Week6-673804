# DSA 2050 - Week 6 Guided Practical Lab: Regression and Baseline Comparison

**Student Name:** Victor Apamo
**Student ID:** 673804


## Lab Objective
This lab builds and evaluates regression models that estimate `Sales_KES` from business characteristics (advertising spend, store visits, discount percentage, region, customer segment and product category) using a 320-row sales dataset. It compares simple baselines against Linear Regression, Ridge Regression and a shallow Decision Tree using MAE, RMSE and R², with leakage-safe preprocessing pipelines, training-only 5-fold cross-validation, residual analysis, customer-segment error checks and overfitting checks, and ends with a deployment recommendation and limitations.

## Models Compared
- Mean baseline (`DummyRegressor`, strategy = mean)
- Median baseline (`DummyRegressor`, strategy = median)
- Linear Regression
- Ridge Regression (alpha = 1.0)
- Decision Tree (max_depth = 4, min_samples_leaf = 8)
- Extra: unrestricted Decision Tree (to demonstrate overfitting)

## Results (20% hold-out test set, `random_state = 42`)
| Model | MAE (KES) | RMSE (KES) | R² | CV MAE (KES) |
|---|---:|---:|---:|---:|
| Mean Baseline | 39,197 | 48,716 | -0.008 | n/a |
| Median Baseline | 39,147 | 48,649 | -0.006 | n/a |
| Linear Regression | 13,758 | 17,470 | 0.870 | 15,977 |
| **Ridge Regression** | **13,714** | **17,459** | **0.870** | **15,971** |
| Decision Tree (depth 4) | 23,776 | 28,768 | 0.648 | 24,498 |

**Best model:** Ridge Regression
**Selected model MAE:** about KES 13,714 on the test set (about 8% of average sales; cross-validated MAE about KES 15,971)

## Two Key Findings
1. **The model clearly beats the baseline.** Ridge Regression cuts the average error by about 65% compared with the median baseline (MAE 13,714 vs 39,147) and explains about 87% of the variation in sales. `StoreVisits` is the most important predictor: removing it pushes MAE up to about KES 34,428, while removing `AdvertisingSpend_KES` pushes it to about KES 21,432.
2. **Errors are largest in low-traffic cases and for Retail.** The model over-predicts sales when store visits are low (MAE about KES 19,300 for rows with fewer than 300 visits, versus about KES 10,700 otherwise), and Retail has the highest segment MAE (about KES 15,561) compared with SME (about KES 10,481). An unrestricted decision tree overfits badly (training MAE 0 vs test MAE about 26,800).

## Three Limitations
1. **Small dataset:** only 320 rows (64 in the test set), so the results are noisy and segment-level findings rest on very few observations.
2. **Teaching-style data and limited predictors:** the data is clean and has only six variables, with no seasonality, store size, competitor, stock or economic information, so real-world accuracy would probably be lower.
3. **Prediction is not causation, and no time check:** the random split does not test future drift, and the model shows association only, so it should not be used alone to justify advertising or traffic decisions.

## Repository Structure
```
DSA2050-Week6-StudentID/
├── DSA2050_Week6_Guided_Regression_Lab.ipynb   # full lab: every Part restated, code, outputs and answers
├── DSA2050_Week6_Sales_Data.csv                # dataset
├── requirements.txt
└── README.md
```

## How to Run
```bash
pip install -r requirements.txt
jupyter notebook DSA2050_Week6_Guided_Regression_Lab.ipynb
```
Then choose **Kernel > Restart & Run All**. The notebook reads `DSA2050_Week6_Sales_Data.csv` from the same folder.
