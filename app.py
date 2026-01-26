import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests
import base64
from pathlib import Path
from geopy.distance import geodesic

st.write("✅ Streamlit app is loading correctly")


# ----------------------------
# Background Image + Overlay
# ----------------------------
def set_background(image_path: str):
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
            background-position: center;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


def add_overlay(opacity: float = 0.75):
    st.markdown(
        f"""
        <style>
        .stApp::before {{
            content: "";
            position: fixed;
            top: 0; left: 0;
            width: 100%; height: 100%;
            background: rgba(255, 255, 255, {opacity});
            z-index: 0;
        }}

        .stApp > div {{
            position: relative;
            z-index: 1;
        }}

        div.stButton > button {{
            width: 100%;
            border-radius: 14px;
            padding: 0.6rem;
            font-weight: 700;
            font-size: 16px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(page_title="ETA Prediction", layout="centered")
st.title("🚚 Delivery ETA Prediction App")
st.caption("Address → Distance → Feature Engineering → ETA + SLA promise + Risk Tag")

# ✅ Background optional
try:
    set_background("assets/bg.jpg")
    add_overlay(opacity=0.75)
except Exception:
    pass


# ----------------------------
# Load Artifacts (cached)
# ----------------------------
@st.cache_resource
def load_artifacts():
    ARTIFACTS = Path("artifacts")

    model = pickle.load(open(ARTIFACTS / "model_v1.pkl", "rb"))
    scaler = pickle.load(open(ARTIFACTS / "scaler_v1.pkl", "rb"))
    feature_list = pickle.load(open(ARTIFACTS / "features_v1.pkl", "rb"))

    ohe = pickle.load(open(ARTIFACTS / "onehot_v1.pkl", "rb"))
    ord_enc = pickle.load(open(ARTIFACTS / "ordinal_encoder_v1.pkl", "rb"))

    city_freq_map = pickle.load(open(ARTIFACTS / "city_freq_map_v1.pkl", "rb"))
    city_freq_mean = pickle.load(open(ARTIFACTS / "city_freq_mean_v1.pkl", "rb"))

    SLA = pickle.load(open(ARTIFACTS / "sla_v1.pkl", "rb"))

    return model, scaler, feature_list, ohe, ord_enc, city_freq_map, city_freq_mean, SLA


model, scaler, feature_list, ohe, ord_enc, city_freq_map, city_freq_mean, SLA = load_artifacts()


# ----------------------------
# Geocoding (OpenCage)
# ----------------------------
@st.cache_data(show_spinner=False)
def get_lat_lon_opencage(address: str, api_key: str):
    url = "https://api.opencagedata.com/geocode/v1/json"
    params = {"q": address, "key": api_key, "limit": 1}

    r = requests.get(url, params=params, timeout=10)
    data = r.json()

    if "results" not in data or len(data["results"]) == 0:
        return None, None

    lat = data["results"][0]["geometry"]["lat"]
    lon = data["results"][0]["geometry"]["lng"]
    return lat, lon


# ----------------------------
# Feature Engineering Functions (from your notebook)
# ----------------------------
def extract_date_features(data):
    data["day"] = data["Order_Date"].dt.day
    data["month"] = data["Order_Date"].dt.month
    data["quarter"] = data["Order_Date"].dt.quarter
    data["year"] = data["Order_Date"].dt.year

    data["day_of_week"] = data["Order_Date"].dt.day_of_week.astype(int)

    data["is_month_start"] = data["Order_Date"].dt.is_month_start.astype(int)
    data["is_month_end"] = data["Order_Date"].dt.is_month_end.astype(int)

    data["is_quarter_start"] = data["Order_Date"].dt.is_quarter_start.astype(int)
    data["is_quarter_end"] = data["Order_Date"].dt.is_quarter_end.astype(int)

    data["is_year_start"] = data["Order_Date"].dt.is_year_start.astype(int)
    data["is_year_end"] = data["Order_Date"].dt.is_year_end.astype(int)

    data["is_weekend"] = np.where(data["day_of_week"].isin([5, 6]), 1, 0)

    return data


def calculate_time_diff(df):
    df["Time_Order_picked_formatted"] = pd.to_datetime(
        df["Order_Date"].astype(str) + " " + df["Time_Order_picked"].astype(str),
        errors="coerce"
    )

    df["Time_Ordered_formatted"] = pd.to_datetime(
        df["Order_Date"].astype(str) + " " + df["Time_Orderd"].astype(str),
        errors="coerce"
    )

    df["Time_Order_picked_formatted"] = df["Time_Order_picked_formatted"] + pd.to_timedelta(
        (df["Time_Order_picked_formatted"] < df["Time_Ordered_formatted"])
        .fillna(False)
        .astype(int),
        unit="D"
    )

    df["order_prepare_time"] = (
        (df["Time_Order_picked_formatted"] - df["Time_Ordered_formatted"])
        .dt.total_seconds() / 60
    )

    # Streamlit has only 1 record, so safest fill
    df["order_prepare_time"] = df["order_prepare_time"].fillna(df["order_prepare_time"].median())

    df.drop(
        ["Time_Orderd", "Time_Order_picked", "Time_Ordered_formatted", "Time_Order_picked_formatted"],
        axis=1,
        inplace=True
    )

    return df


# ----------------------------
# Preprocessing (Inference)
# ----------------------------
scale_cols = [
    'Delivery_person_Age',
    'Delivery_person_Ratings',
    'Restaurant_latitude',
    'Restaurant_longitude',
    'Delivery_location_latitude',
    'Delivery_location_longitude',
    'order_prepare_time',
    'distance',
    'City_code_freq',
    'Vehicle_condition',
    'multiple_deliveries',
    'Road_traffic_density',
    'day_of_week'
]

def preprocess_input(raw_df: pd.DataFrame):
    df = raw_df.copy()

    # ✅ City_code frequency encoding (IMPORTANT)
    df["City_code_freq"] = df["City_code"].map(city_freq_map).fillna(city_freq_mean)
    df.drop(columns=["City_code"], inplace=True)

    # ✅ Ordinal encoding: Road_traffic_density
    df[["Road_traffic_density"]] = ord_enc.transform(df[["Road_traffic_density"]])

    # ✅ One-hot encoding (includes City type)
    onehot_cols = ["Weather_conditions", "Type_of_order", "Type_of_vehicle", "Festival", "City"]
    onehot_cols = [c for c in onehot_cols if c in df.columns]

    ohe_array = ohe.transform(df[onehot_cols])
    ohe_df = pd.DataFrame(
        ohe_array,
        columns=ohe.get_feature_names_out(onehot_cols),
        index=df.index
    )

    df.drop(columns=onehot_cols, inplace=True)
    df = pd.concat([df, ohe_df], axis=1)

    # ✅ Align with training feature_list
    df = df.reindex(columns=feature_list, fill_value=0)

    # ✅ Scale only those columns which were scaled in training
    cols_to_scale = [c for c in scale_cols if c in df.columns]

    df_scaled = df.copy()
    df_scaled[cols_to_scale] = scaler.transform(df_scaled[cols_to_scale])

    return df_scaled
    
    


# ----------------------------
# UI Inputs
# ----------------------------
st.subheader("1) Address Inputs")
restaurant_address = st.text_input("Restaurant Address", placeholder="Example: Gachibowli, Hyderabad")
delivery_address = st.text_input("Delivery Address", placeholder="Example: Hitech City, Hyderabad")

st.subheader("2) Rider / Order Inputs")
c1, c2 = st.columns(2)

with c1:
    Delivery_person_Age = st.slider("Delivery Person Age", 18, 60, 30)
    Delivery_person_Ratings = st.slider("Delivery Person Ratings", 1.0, 5.0, 4.2, step=0.1)
    Vehicle_condition = st.selectbox("Vehicle Condition", [0, 1, 2, 3])
    multiple_deliveries = st.selectbox("Multiple Deliveries", [0, 1, 2, 3])

with c2:
    # ✅ City_code dropdown for frequency encoding
    City_code = st.selectbox("City Code", sorted(list(city_freq_map.keys())))

    # ✅ City type for one-hot
    City = st.selectbox("City Type", ["Urban ", "Semi-Urban ", "Rural "])

    Road_traffic_density = st.selectbox("Road Traffic Density", ["Low ", "Medium ", "High ", "Jam "])
    Festival = st.selectbox("Festival", ["Yes ", "No "])

st.subheader("3) Order Context Inputs")
c3, c4 = st.columns(2)

with c3:
    Weather_conditions = st.selectbox(
        "Weather Conditions",
        ["Fog", "Sandstorms", "Stormy", "Sunny", "Windy"]
    )

with c4:
    Type_of_order = st.selectbox("Type of Order", ["Drinks ", "Meal ", "Snack "])
    Type_of_vehicle = st.selectbox(
        "Type of Vehicle",
        ["electric_scooter ", "motorcycle ", "scooter "]
    )

st.subheader("4) Time Inputs (for order_prepare_time)")
order_date = st.date_input("Order Date", pd.Timestamp.today().date())
Time_Orderd = st.text_input("Order Time (HH:MM:SS)", value="12:00:00")
Time_Order_picked = st.text_input("Picked Time (HH:MM:SS)", value="12:15:00")


# ----------------------------
# Predict
# ----------------------------
st.subheader("5) Prediction")

if st.button("Predict ETA"):

    # ✅ API key safety
    if "OPENCAGE_API_KEY" not in st.secrets:
        st.error("OpenCage API key missing. Add it to `.streamlit/secrets.toml`.")
        st.stop()

    api_key = st.secrets["OPENCAGE_API_KEY"]

    if not restaurant_address or not delivery_address:
        st.warning("Please enter both addresses.")
        st.stop()

    with st.spinner("Fetching coordinates and computing distance..."):
        lat1, lon1 = get_lat_lon_opencage(restaurant_address, api_key)
        lat2, lon2 = get_lat_lon_opencage(delivery_address, api_key)

        if lat1 is None or lat2 is None:
            st.error("Could not fetch location for one of the addresses. Try a more specific address.")
            st.stop()

        distance = geodesic((lat1, lon1), (lat2, lon2)).km

    st.info(f"📍 Auto-calculated Distance: **{distance:.2f} km**")

    # ✅ Build raw input row
    input_df = pd.DataFrame([{
        "Delivery_person_Age": Delivery_person_Age,
        "Delivery_person_Ratings": Delivery_person_Ratings,
        "Restaurant_latitude": lat1,
        "Restaurant_longitude": lon1,
        "Delivery_location_latitude": lat2,
        "Delivery_location_longitude": lon2,
        "Road_traffic_density": Road_traffic_density,
        "Vehicle_condition": Vehicle_condition,
        "multiple_deliveries": multiple_deliveries,
        "Order_Date": pd.to_datetime(order_date),
        "Time_Orderd": Time_Orderd,
        "Time_Order_picked": Time_Order_picked,
        "order_prepare_time": np.nan,  # will be created by calculate_time_diff
        "distance": distance,
        "City_code": City_code,
        "City": City,
        "Festival": Festival,
        "Weather_conditions": Weather_conditions,
        "Type_of_order": Type_of_order,
        "Type_of_vehicle": Type_of_vehicle
    }])

    # ✅ Feature engineering (same as training)
    input_df["Order_Date"] = pd.to_datetime(input_df["Order_Date"])
    input_df = extract_date_features(input_df)
    input_df = calculate_time_diff(input_df)

    # ✅ Predict
    try:
        X_input = preprocess_input(input_df)
        eta_pred = model.predict(X_input)[0]

        promised_time = eta_pred + SLA

        st.divider()
        st.subheader("✅ Final Output")

        st.write(f"🕒 **Predicted ETA:** `{eta_pred:.2f} minutes`")
        st.write(f"🧾 **SLA Buffer:** `+{SLA} minutes`")
        st.write(f"🎯 **Promised Delivery Time (ETA + SLA):** `{promised_time:.2f} minutes`")

        # ✅ Risk Tag
        if eta_pred <= 20:
            st.warning("⚠️ **High Risk:** Short ETA predictions have higher underestimation risk.")
        elif 20 < eta_pred <= 35:
            st.info("✅ **Moderate Risk:** Prediction is in a stable ETA range.")
        else:
            st.success("🟢 **Low Risk:** Model is conservative in high ETA ranges.")

        st.caption("Note: SLA breach can be confirmed only after actual delivery time is available.")

    except Exception as e:
        st.error("❌ Error during preprocessing/prediction (check encoders, feature_list, column names).")
        st.exception(e)