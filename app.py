import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

st.set_page_config(
    page_title="AA DFW Delay Predictor",
    page_icon="✈",
    layout="wide"
)

# ── load data & train models once ────────────────────────────────────────────
@st.cache_resource(show_spinner="Training models on historical data…")
def load_models():
    df = pd.read_csv("modeling_dataset.csv")

    feature_cols = [c for c in df.columns if c not in [
        "LEG_ARVL_VARNCE_MIN_QTY", "CHANGE_FLAG_1MIN", "CHANGE_FLAG_10MIN"
    ]]

    X = df[feature_cols]
    y = df["CHANGE_FLAG_10MIN"]
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    rf = RandomForestClassifier(
        n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)

    df2 = df[df["CHANGE_FLAG_10MIN"] == 1].copy()
    X2 = df2[feature_cols]
    y2 = df2["LEG_ARVL_VARNCE_MIN_QTY"]
    X2_train, _, y2_train, _ = train_test_split(X2, y2, test_size=0.2, random_state=42)

    xgb2 = XGBRegressor(random_state=42, n_jobs=-1)
    xgb2.fit(X2_train, y2_train)

    medians = df[feature_cols].median().to_dict()
    return rf, xgb2, feature_cols, medians

rf, xgb2, feature_cols, medians = load_models()

# ── mappings ──────────────────────────────────────────────────────────────────
AIRLINE_MAP   = {"AA (American Airlines)": 0, "MQ (Envoy/Eagle)": 1,
                 "OH (PSA Airlines)": 2,     "OO (SkyWest)": 3, "YV (Mesa Air)": 4}
FLEET_MAP     = {"A319": 0, "A320": 1, "A321": 2, "B737": 3, "B777": 4,
                 "B787": 5, "CRJ":  6, "EMJ":  7, "ERJ":  8, "M80":  9}
DAY_MAP       = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
                 "Friday": 4, "Saturday": 5, "Sunday": 6}
MONTH_MAP     = {"Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4,  "May": 5,  "Jun": 6,
                 "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12}

# ── page header ───────────────────────────────────────────────────────────────
st.title("✈  American Airlines — DFW Departure Delay Predictor")
st.caption("Group 14 · ADTA 5940 · UNT Spring 2026 · Pre-departure prediction only")
st.divider()

# ── sidebar inputs ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙  Flight Parameters")

    st.subheader("Departure Info")
    dep_variance = st.slider(
        "Departure delay so far (minutes)", 0, 488, 0,
        help="How many minutes late the flight already is at the gate"
    )
    airline = st.selectbox("Airline", list(AIRLINE_MAP.keys()))
    fleet   = st.selectbox("Aircraft type", list(FLEET_MAP.keys()), index=3)
    duration = st.slider("Scheduled flight duration (minutes)", 30, 337, 120)

    st.subheader("When")
    hour    = st.slider("Departure hour (0 = midnight, 14 = 2 PM)", 0, 23, 14)
    day     = st.selectbox("Day of week", list(DAY_MAP.keys()), index=0)
    month   = st.selectbox("Month", list(MONTH_MAP.keys()), index=6)

    st.subheader("Weather at DFW (Departure)")
    dep_wind    = st.slider("Wind speed at DFW (knots)", 0, 40, 9)
    dep_vis     = st.slider("Visibility at DFW (miles)", 0.0, 10.0, 10.0, step=0.5)
    dep_precip  = st.slider("Precipitation at DFW (inches)", 0.0, 2.0, 0.0, step=0.05)
    wx_flag     = st.checkbox("Official AA weather delay flag", value=False)

    st.subheader("Weather at Destination")
    arvl_wind   = st.slider("Wind speed at destination (knots)", 0, 70, 7)
    arvl_vis    = st.slider("Visibility at destination (miles)", 0.0, 11.0, 10.0, step=0.5)
    arvl_precip = st.slider("Precipitation at destination (inches)", 0.0, 2.0, 0.0, step=0.05)

    predict_btn = st.button("🔍  Predict", use_container_width=True, type="primary")

# ── build feature vector ───────────────────────────────────────────────────────
def make_input():
    row = medians.copy()
    row["LEG_DEP_VARNCE_MIN_QTY"]      = dep_variance
    row["dep_hour"]                     = hour
    row["dep_dayofweek"]                = DAY_MAP[day]
    row["dep_month"]                    = MONTH_MAP[month]
    row["flight_duration_min"]          = duration
    row["OPERAT_AIRLN_IATA_CD_ENC"]     = AIRLINE_MAP[airline]
    row["SCHD_FLEET_CD_ENC"]            = FLEET_MAP[fleet]
    row["ACTL_FLEET_CD_ENC"]            = FLEET_MAP[fleet]
    row["dep_wind_speed"]               = dep_wind
    row["dep_visibility"]               = dep_vis
    row["dep_precipitation"]            = dep_precip
    row["weather_delay_flag"]           = int(wx_flag)
    row["arvl_wind_speed"]              = arvl_wind
    row["arvl_visibility"]              = arvl_vis
    row["arvl_precipitation"]           = arvl_precip
    return pd.DataFrame([row])[feature_cols]

# ── main area ─────────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1.1, 1])

with col_left:
    st.subheader("Your Flight Setup")
    summary = {
        "Airline":              airline,
        "Aircraft":             fleet,
        "Flight duration":      f"{duration} min",
        "Departs":              f"{hour:02d}:00  {day}  ({month})",
        "Gate delay so far":    f"{dep_variance} min",
        "DFW wind / vis / rain":f"{dep_wind} kts / {dep_vis} mi / {dep_precip} in",
        "Dest wind / vis / rain":f"{arvl_wind} kts / {arvl_vis} mi / {arvl_precip} in",
        "AA weather flag":      "YES" if wx_flag else "NO",
    }
    st.table(pd.DataFrame(summary.items(), columns=["Parameter", "Value"]))

with col_right:
    st.subheader("Prediction")

    if not predict_btn:
        st.info("Set the parameters on the left, then click **Predict**.")
    else:
        X_new = make_input()

        # Stage 1 — will it be late?
        prob_late = rf.predict_proba(X_new)[0][1]
        is_late   = prob_late >= 0.5

        if is_late:
            # Stage 2 — how many minutes?
            delay_min = float(xgb2.predict(X_new)[0])
            delay_min = max(10.0, delay_min)   # stage 2 only applies when 10+ late

            st.error("🔴  This flight is predicted to arrive **LATE**")

            c1, c2 = st.columns(2)
            c1.metric("Predicted delay", f"{delay_min:.0f} min")
            c2.metric("Late probability", f"{prob_late*100:.0f}%")

            # severity band
            if delay_min < 30:
                band, color = "Minor delay (< 30 min)", "orange"
            elif delay_min < 60:
                band, color = "Moderate delay (30–60 min)", "red"
            else:
                band, color = "Severe delay (60+ min)", "darkred"
            st.markdown(f"**Severity:** :{color}[{band}]")

            st.progress(min(prob_late, 1.0))

            st.divider()
            st.markdown("**What's driving this prediction?**")
            drivers = []
            if dep_variance > 15:
                drivers.append(f"• Flight already {dep_variance} min late at the gate")
            if dep_wind > 15 or dep_precip > 0 or dep_vis < 5:
                drivers.append("• Poor weather conditions at DFW")
            if arvl_wind > 20 or arvl_precip > 0 or arvl_vis < 5:
                drivers.append("• Adverse weather at destination")
            if wx_flag:
                drivers.append("• Official AA weather delay flag is active")
            if hour in range(7, 10) or hour in range(15, 19):
                drivers.append("• Peak hour traffic congestion window")
            if not drivers:
                drivers.append("• Combination of operational factors")
            for d in drivers:
                st.write(d)

        else:
            st.success("🟢  This flight is predicted to arrive **ON TIME**")

            c1, c2 = st.columns(2)
            c1.metric("On-time probability", f"{(1-prob_late)*100:.0f}%")
            c2.metric("Late probability", f"{prob_late*100:.0f}%")

            st.progress(1 - prob_late)
            st.caption("Stage 2 (delay magnitude) not needed — flight expected on time.")

# ── footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Stage 1: Random Forest classifier (Accuracy 89.3%, AUC-ROC 94.9%)  •  "
    "Stage 2: XGBoost regressor (MAE ±9.9 min, R² 0.966)  •  "
    "Trained on 211,800 DFW departures (2021–2024)"
)
