# Airbnb Revenue Optimization
### Leveraging Machine Learning & Market Segmentation to Maximize Host Revenue

> **Purdue University | MS Business Analytics & Information Management**

---

## Project Overview

Airbnb hosts face a persistent challenge: how do you maximize revenue and occupancy in an increasingly competitive marketplace? This project applies end-to-end data science — from exploratory analysis and clustering to predictive modeling and time series forecasting — to a dataset of **39,150 Airbnb listings** to uncover what actually drives host performance and revenue.

The analysis identifies four distinct market segments, quantifies the financial impact of Superhost status, and builds a predictive model to help hosts anticipate and improve their standing.

---

## Business Problem

- Hosts struggle to identify which factors — pricing, availability, amenities, reviews — most influence their revenue
- Superhost status is widely perceived as valuable, but its actual financial impact was unquantified
- Seasonal demand swings make occupancy planning difficult without data-driven forecasting
- There was no systematic framework for segmenting listings by performance profile and tailoring strategies accordingly

**Goal:** Build a data-driven framework to segment the Airbnb market, identify key revenue drivers, quantify the Superhost premium, and predict which hosts are likely to achieve Superhost status.

---

## Dataset

| Attribute | Detail |
|---|---|
| Source | Airbnb Superhost Evaluation Data (public) |
| Size | 39,150 listings, 111 features |
| Time Span | April 2017 – 2018, July 2016–2017, October 2016–2017, January 2017–2018 |
| Key Features | Nightly rate, occupancy rate, revenue, Superhost status, ratings, reviews, availability, geographic data |

**Key variables analyzed:** `revenue`, `occupancy_rate`, `host_is_superhost_in_period`, `nightly_rate`, `rating_ave_pastYear`, `numReviews_pastYear`, `available_days`, `booked_days`

---

## 🔧 Methodology

### 1. Data Preprocessing
- **Missing value imputation:** Median for numerical columns; mode or `'Unknown'` category for categorical columns
- **Outlier handling:** Winsorization at 1st and 99th percentiles for revenue, occupancy rate, ratings, and review counts — preserving high-revenue outliers while reducing noise
- **Column removal:** Dropped `Integrated Property Manager` (100% missing values)
- **Result:** Clean dataset of 39,150 rows × 110 columns, zero duplicates

### 2. Exploratory Data Analysis (EDA)
- Analyzed Superhost status distribution (≈25% of listings are Superhosts)
- Visualized revenue trends over time, seasonal patterns, and top-performing neighborhoods
- Georgetown, Downtown/Penn Quarter, and Judiciary Square emerged as highest average revenue neighborhoods

### 3. Market Segmentation — K-Means Clustering
- Applied **Elbow Method** (KElbowVisualizer) to determine optimal cluster count → **k = 4**
- Performed K-Means clustering on pricing, availability, Superhost status, revenue, and occupancy features

| Cluster | Profile | Avg Nightly Rate | Avg Revenue | Avg Occupancy |
|---|---|---|---|---|
| Cluster 0 | Budget-Friendly | $162 | $2,165 | 24.6% |
| Cluster 1 | Superhost-Managed | $163 | $3,472 | 21.3% |
| Cluster 2 | High Availability, Non-Superhosts | $175 | $2,781 | 15.8% |
| Cluster 3 | Luxury Listings | $941 | $3,841 | 16.0% |

**Key finding:** Superhost-managed listings (Cluster 1) outperform similarly priced non-Superhost listings across every revenue and satisfaction metric.

### 4. Hypothesis Testing — The Superhost Premium
Conducted independent samples **t-tests** to quantify the financial impact of Superhost status:

| Metric | Superhosts | Non-Superhosts | Difference | p-value |
|---|---|---|---|---|
| Average Revenue | $3,457.70 | $2,666.84 | **+$838.81** | 0.0000 |
| Average Occupancy Rate | 20.30% | 16.21% | **+3.54 pp** | 0.0000 |

**Result:** Superhost status is associated with statistically significant gains in both revenue and occupancy. The differences are not due to chance.

### 5. Revenue Driver Analysis — Feature Importance by Cluster
Used **Random Forest** feature importance to identify the top revenue drivers within each cluster:
- **Occupancy rate** is the #1 revenue driver across all four clusters
- Revenue increases with occupancy up to ~60% — beyond that, other factors (pricing, Superhost status, amenities) become the differentiating levers
- Nightly rate and available days are consistently the 2nd and 3rd most important features

### 6. Predictive Modeling — Superhost Status Classification
Built classification models to predict whether a host will achieve Superhost status in the next evaluation period:

| Model | Accuracy | Precision | Recall | AUC-ROC |
|---|---|---|---|---|
| **LightGBM** ✅ | 96.4% | 61.1% | 17.9% | **0.960** |
| CatBoost | 96.4% | 57.9% | 17.2% | 0.959 |
| XGBoost | 96.3% | 53.4% | 20.9% | 0.956 |
| Random Forest | 96.5% | 65.1% | 18.8% | 0.951 |
| Logistic Regression | 96.2% | 40.0% | 0.4% | 0.929 |

**Selected model:** LightGBM — highest AUC-ROC (0.960), optimized for large datasets with imbalanced classes, and provides interpretable feature importance.

**Top predictors of Superhost attainment:** Number of reviews, average rating, nightly rate, available days, revenue, occupancy rate.

### 7. Time Series Forecasting — ARIMA
- Modeled occupancy rate trends over time using **ARIMA**
- Identified strong seasonal patterns: peaks in January and July
- Declining peak revenue trend over 2017–2020 suggests increasing market saturation

---

## 📊 Key Findings

1. **Superhost status earns ~$839 more per evaluation period** on average — a statistically significant premium driven by trust signals, better reviews, and higher booking rates

2. **Occupancy rate is the single most powerful revenue lever** across all listing types — optimizing toward ~60% occupancy should be the primary goal for most hosts

3. **Market segmentation reveals four distinct strategies** — budget listings need availability improvements, Superhost listings benefit from dynamic pricing, luxury listings should focus on exclusivity marketing

4. **LightGBM predicts Superhost attainment with 96% accuracy** — reviews and ratings are the most actionable features hosts can influence

5. **Georgetown and Downtown DC neighborhoods** generate the highest average revenue, suggesting geographic positioning matters significantly

---

## Strategic Recommendations

| Segment | Strategy |
|---|---|
| Budget-Friendly (Cluster 0) | Increase availability, improve ratings, target price-sensitive travelers |
| Superhost-Managed (Cluster 1) | Implement dynamic pricing to capture seasonal peaks |
| High Availability (Cluster 2) | Pursue Superhost status; use competitive pricing |
| Luxury (Cluster 3) | Target affluent travelers with tailored marketing; highlight exclusivity |

**For all hosts:** Prioritize review volume and rating quality — these are the strongest predictors of Superhost attainment and the most actionable levers for revenue growth.

---

## Tools & Technologies

`Python` `pandas` `scikit-learn` `LightGBM` `XGBoost` `CatBoost` `statsmodels` `matplotlib` `seaborn` `yellowbrick`

**Methods:** K-Means Clustering, Elbow Method, Winsorization, Hypothesis Testing (t-test), Random Forest Feature Importance, LightGBM Classification, ARIMA Forecasting

---

## Repository Structure

```
airbnb-revenue-optimization/
│
├── notebooks/
│   └── airbnb_revenue_optimization.ipynb    # Full analysis notebook
│
├── presentation/
│   └── AIBD_Team_Project_Team38.pdf         # Final project presentation
│
├── data/
│   └── README_data.md                       # Data source description
│                                            # (raw data not included — see source below)
│
└── README.md
```

**Data source:** Airbnb Superhost Evaluation Dataset — publicly available via Inside Airbnb and academic repositories. Eight evaluation periods from 2016–2018.

---

## Context

This project was completed as part of the **AI for Business Decisions (AIBD)** course in the MS Business Analytics & Information Management program at **Purdue University, Daniels School of Business**.

---

## Author

**Vandana Brahmasa**
[LinkedIn](https://www.linkedin.com/in/vandana-brahmasa) | [vandanabrahmasa92@gmail.com](mailto:vandanabrahmasa92@gmail.com)
