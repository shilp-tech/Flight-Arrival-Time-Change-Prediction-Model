# Flight Arrival Time Change Prediction Model
### American Airlines — DFW Departure Delay Predictor

**Course:** ADTA 5940 · University of North Texas · Spring 2026  
**Group 14:** Shilp Patel  
**Scope:** Pre-departure prediction of arrival delay for flights departing DFW

---

## Demo

<video src="app_view.mov" controls width="100%"></video>

---

## Overview

This project builds a **two-stage machine learning pipeline** to predict whether an American Airlines flight departing Dallas/Fort Worth (DFW) will arrive 10 or more minutes late — and if so, by how many minutes. The prediction is made entirely from pre-departure information, making it useful for operational decisions like Connect Assist holds before a flight pushes back.

The final model is deployed as an interactive **Streamlit web app** where airline operations staff can input flight parameters and receive an instant delay prediction with severity classification.

---

## The Problem

American Airlines operates Connect Assist, a system that decides whether to hold a connecting flight at DFW when an inbound feeder flight is running late. DFW's minimum connection time is 40–45 minutes. A 10-minute late arrival consumes 25% of that buffer. The current system relies on manually posted ETA updates (FLIFO), which can lag real conditions.

**Goal:** Predict, before departure, whether a flight will arrive 10+ minutes late so Connect Assist can act earlier with more confidence.

---

## Two-Stage Architecture

```
Flight Parameters + Weather + Airport Data
               │
               ▼
┌──────────────────────────────────────┐
│  Stage 1 — Random Forest Classifier │
│  Will this flight arrive 10+ min    │
│  late? (Binary: Yes / No)           │
│                                      │
│  Accuracy : 89.3%                    │
│  AUC-ROC  : 94.9%                    │
└──────────────────┬───────────────────┘
                   │ If YES
                   ▼
┌──────────────────────────────────────┐
│  Stage 2 — XGBoost Regressor        │
│  How many minutes late?             │
│                                      │
│  MAE  : ±9.9 minutes                 │
│  R²   : 0.966                        │
└──────────────────────────────────────┘
```

---

## Repository Structure

```
.
├── 01_airport_cleaning.ipynb      # Step 1 — Clean airport reference data
├── 02_eta_cleaning.ipynb          # Step 2 — Clean raw ETA/FLIFO flight data
├── 03_join_and_combine.ipynb      # Step 3 — Join flights + airports + weather
├── 04_feature_eng.ipynb           # Step 4 — Feature engineering & target creation
├── 05_modeling.ipynb              # Step 5 — Model training & evaluation
├── 00b_EDA_combined.ipynb         # EDA on the combined dataset
├── 00c_EDA_modeling.ipynb         # Pre-modeling checklist & sanity checks
├── app.py                         # Streamlit prediction app
├── app_view.mov                   # Demo screen recording of the app
├── Final Research Paper.docx      # Full academic write-up
└── Group_14_Shilp_Patel_Flight Arrival Time Change Prediction Model (1).pptx
```

---

## Data Pipeline

The pipeline processes three raw data sources through five sequential notebooks.

### Step 1 — Airport Cleaning (`01_airport_cleaning.ipynb`)
- Loaded `AIRPORT_INFO_backup.csv` — AA's internal airport reference table
- Converted raw lat/lon from degree/minute/second format to decimal
- Manually corrected zero-length runway entries for 4 airports (OAI, AMT, IRP, KAG) using official aviation records
- Dropped audit columns (`STATE_PROVNC_CD`, `SF_LOAD_TMS`)
- Output: `AIRPORT_INFO_clean.csv`

### Step 2 — ETA Cleaning (`02_eta_cleaning.ipynb`)
- Loaded `ETA_CHANGE_PREDICTION_backup.csv` — AA's FLIFO ETA update log (~2.26M rows)
- Removed cancelled flights and diverted flights (actual airport ≠ scheduled airport)
- Filtered to valid AA regional partner airlines only: AA, MQ, OH, OO, YV (removed YX)
- Validated flight numbers to 4-digit range (1000–9999)
- Calculated `flight_duration_min` from scheduled departure and arrival timestamps
- Removed impossible durations (< 1 min or > 1,440 min)
- Dropped data-leakage columns (fields only known after landing), audit fields, and redundant timestamps
- Output: `ETA_CHANGE_PREDICTION_clean.csv`

### Step 3 — Join & Combine (`03_join_and_combine.ipynb`)
- Filtered to US domestic arrivals only (NOAA weather coverage)
- Left-joined cleaned airport info twice: once for departure airport (prefix `dep_`), once for arrival (prefix `arvl_`)
- Loaded `weather_hourly.csv` — NOAA hourly observations for DFW and all US arrival airports
- Joined departure weather by rounding scheduled departure GMT timestamp to the nearest hour
- Joined arrival weather by rounding scheduled arrival GMT timestamp to the nearest hour
- Output: `combined_dataset.csv` (~2.06M rows)

### Step 4 — Feature Engineering (`04_feature_eng.ipynb`)
- Kept only the **last pre-departure FLIFO update** per flight per day (highest `POST_NBR` where `MINS_TO_SCHD_DEP_QTY ≤ 0`) — this is the last snapshot of operational knowledge before pushback
- Final dataset: **211,800 DFW departures (2021–2024)**
- Extracted time features: `dep_hour`, `dep_dayofweek`, `dep_month`
- Created `weather_delay_flag` binary from AA's FLIFO delay reason codes (MTR, WXL, WXD, WXS, etc.)
- Created targets:
  - `CHANGE_FLAG_1MIN` — arrival changed by ≥ 1 minute (98% positive — too imbalanced for use)
  - `CHANGE_FLAG_10MIN` — arrival ≥ 10 minutes late (80% positive — used for Stage 1)
  - `LEG_ARVL_VARNCE_MIN_QTY` — actual arrival delay in minutes (Stage 2 regression target)
- Label-encoded categorical columns: airline, fleet type, departure status, arrival status, destination airport, weather codes
- Filled missing DFW weather with medians; dropped rows with missing arrival airport weather
- Capped outliers at 1st / 99th percentile for variance targets
- Output: `modeling_dataset.csv` (35 columns, zero missing values)

### Step 5 — Modeling (`05_modeling.ipynb`)
- **Stage 1 — Classification (predict 10+ min late)**
  - Logistic Regression (baseline with `StandardScaler`)
  - Random Forest with `class_weight='balanced'` ← **selected model**
  - XGBoost with `scale_pos_weight` for imbalance handling
- **Stage 2 — Regression (predict exact delay minutes, on late-only subset)**
  - Linear Regression (baseline)
  - Random Forest Regressor
  - XGBoost Regressor ← **selected model**
- Evaluation: stratified 80/20 train/test split; classification scored on AUC-ROC and F1; regression scored on MAE, RMSE, and R²

---

## Features (35 total)

| Category | Features |
|---|---|
| Operational | `LEG_DEP_VARNCE_MIN_QTY` (gate delay so far), `POST_NBR` (# of ETA updates posted), `flight_duration_min` |
| Time | `dep_hour`, `dep_dayofweek`, `dep_month` |
| Airline & Fleet | `OPERAT_AIRLN_IATA_CD_ENC`, `SCHD_FLEET_CD_ENC`, `ACTL_FLEET_CD_ENC` |
| Departure Airport | `dep_airport_latitude`, `dep_airport_longitude`, `dep_LNGST_RUNWAY_FT_QTY`, `dep_ELEVATN_FT_QTY` |
| Arrival Airport | `arvl_airport_latitude`, `arvl_airport_longitude`, `arvl_LNGST_RUNWAY_FT_QTY`, `arvl_ELEVATN_FT_QTY` |
| DFW Weather | `dep_wind_speed`, `dep_visibility`, `dep_precipitation` |
| Destination Weather | `arvl_wind_speed`, `arvl_visibility`, `arvl_precipitation` |
| Flags | `weather_delay_flag`, `SUBSEQUENT_LEG_OF_DIVERT_IND` |
| Status codes | `DEP_STATUS_DESC_ENC`, `ARVL_STATUS_DESC_ENC`, `OP_PRE_STATUS_CD_ENC`, `OP_STATUS_CD_ENC` |
| Destination | `SCHD_LEG_ARVL_AIRPRT_IATA_CD_ENC` |

**Top predictors (by feature importance):** `POST_NBR` (number of ETA updates), `dep_visibility`, `dep_wind_speed`, `dep_precipitation`, `dep_hour`, `flight_duration_min`.

---

## Model Results

### Stage 1 — Random Forest Classifier

| Metric | Score |
|---|---|
| Accuracy | **89.3%** |
| AUC-ROC | **94.9%** |
| Class weight | `balanced` (handles 80/20 imbalance) |
| Training size | 169,440 flights |

### Stage 2 — XGBoost Regressor

| Metric | Score |
|---|---|
| MAE | **±9.9 minutes** |
| R² | **0.966** |
| Applied to | Late flights only (Stage 1 positive predictions) |

---

## Key EDA Findings

- **Departure gate delay is the single strongest predictor** of arrival delay. A flight already 15+ minutes late at the gate almost always arrives late.
- **POST_NBR** (number of FLIFO updates posted) is highly correlated with delay — more updates signal an evolving operational problem.
- **Peak delay hours:** 7–10 AM and 3–7 PM due to traffic congestion windows.
- **Weather impact:** Low visibility (< 3 mi) and precipitation at DFW increase mean delay by 40–80 minutes. Destination weather has a secondary but still significant effect.
- **Seasonal pattern:** Summer months (June–August) and December have the highest late-arrival rates.
- **Airline breakdown:** Regional carriers (MQ, OO) show slightly higher late rates than mainline AA on the same routes.
- **CHANGE_FLAG_1MIN** was tested but discarded — 98:1 class imbalance made it impractical for classification; CHANGE_FLAG_10MIN (80:20) is the operational threshold with real Connect Assist relevance.

---

## Streamlit App (`app.py`)

The app trains both models live on startup (using `@st.cache_resource`) and exposes a sidebar for all input parameters.

**Inputs:**
- Gate delay so far (minutes)
- Airline (AA, MQ, OH, OO, YV)
- Aircraft type (A319, A320, A321, B737, B777, B787, CRJ, EMJ, ERJ, M80)
- Scheduled flight duration
- Departure hour, day of week, month
- DFW weather: wind speed, visibility, precipitation
- AA official weather delay flag
- Destination weather: wind speed, visibility, precipitation

**Output:**
- Stage 1: On time (green) / Late (red) with probability
- Stage 2 (if late): Predicted delay in minutes, severity band (Minor < 30 min / Moderate 30–60 min / Severe 60+ min), progress bar, and plain-English explanation of the top driving factors

**Run locally:**
```bash
pip install streamlit pandas numpy scikit-learn xgboost
# Place modeling_dataset.csv in the same directory
streamlit run app.py
```

> **Note:** `modeling_dataset.csv` is not included in this repository due to American Airlines data policy (see below). The app will not run without it.

---

## Data Availability Notice

> **The raw and processed datasets used in this project cannot be shared publicly.**
>
> All flight operations data (`ETA_CHANGE_PREDICTION_backup.csv`), airport reference data (`AIRPORT_INFO_backup.csv`), and derived datasets (`combined_dataset.csv`, `modeling_dataset.csv`) are proprietary to **American Airlines** and were provided exclusively for academic research under a data use agreement with the University of North Texas. Redistribution of any form of this data — raw, cleaned, or aggregated — is not permitted under American Airlines' data policy.
>
> To reproduce this project, you would need access to:
> - AA's internal FLIFO/ETA update database
> - AA's internal airport reference table
> - NOAA ISD (Integrated Surface Database) hourly weather observations for US airports

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.x | Core language |
| pandas, numpy | Data wrangling |
| scikit-learn | Logistic Regression, Random Forest, preprocessing, metrics |
| XGBoost | Gradient boosting classifier and regressor |
| matplotlib, seaborn | EDA visualizations |
| Streamlit | Interactive prediction web app |

---

## Academic Context

This project was completed as the capstone for **ADTA 5940 — Applied Data Analytics** at the University of North Texas, Spring 2026. The dataset was provided by American Airlines in partnership with UNT's analytics program. All modeling decisions, code, and analysis are original work by Group 14.

For questions about this project, contact: **shilppatel3100@gmail.com**
