# app.py
# ==============================
# E-COMMERCE ANALYTICS DASHBOARD
# ==============================

import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import numpy as np
from pathlib import Path
from utils.style import style_plotly
from dashboard_pages.Overview import show_overview
from dashboard_pages.Customer_Segmentation import (show_customer_segmentation)
from dashboard_pages.RFM_Analysis import show_rfm
from dashboard_pages.Customer_Prediction import show_prediction
# ------------------------------
# PAGE CONFIG
# ------------------------------
st.set_page_config(
    page_title="E-Commerce Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)
PLOTLY_FONT = "#2A2A2A"

PLOTLY_BG = "#FFFFFF"

SEGMENT_COLORS = [
    "#8A6ED7",
    "#5E51E0",
    "#AB9DD7",
    "#D8CFF2",
    "#7B61FF"
]




# ------------------------------
# PATH SETUP
# ------------------------------
APP_DIR = Path(__file__).resolve().parent      # e-com-analysis/app
BASE_DIR = APP_DIR.parent                     # e-com-analysis
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
ASSETS_DIR = BASE_DIR / "assets"

# LOAD CSS
# ------------------------------
def load_css():

    css_file = ASSETS_DIR / "style.css"

    if not css_file.exists():
        st.error(f"CSS file not found:\n{css_file}")
        st.stop()

    with open(css_file, "r", encoding="utf-8") as f:
        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )

load_css()

# ------------------------------
# DATASET UPLOAD
# ------------------------------
uploaded_file = st.sidebar.file_uploader(
    "Upload Retail Dataset",
    type=["csv"]
)
with st.sidebar.expander("ℹ️ Required CSV Format"):
    st.markdown("""
    **Required columns:**
    - Invoice
    - Customer ID
    - InvoiceDate
    - Quantity
    - Price
    - Country
    """)

uploaded_customer_features = None

if uploaded_file is not None:

    uploaded_df = pd.read_csv(uploaded_file)

    st.sidebar.success("Dataset Uploaded")

   

    required_columns = [
        "Invoice",
        "Customer ID",
        "InvoiceDate",
        "Quantity",
        "Price",
        "Country"
    ]

    missing_columns = [
        col
        for col in required_columns
        if col not in uploaded_df.columns
    ]

    if missing_columns:

        st.error(
            f"Missing columns: {missing_columns}"
        )

        st.stop()

    uploaded_df = uploaded_df.dropna(
        subset=["Customer ID"]
    )

    uploaded_df = uploaded_df[
        uploaded_df["Quantity"] > 0
    ]

    uploaded_df = uploaded_df[
        uploaded_df["Price"] > 0
    ]

    uploaded_df["InvoiceDate"] = pd.to_datetime(
        uploaded_df["InvoiceDate"]
    )

    uploaded_df["total_price"] = (
        uploaded_df["Quantity"]
        * uploaded_df["Price"]
    )
    st.session_state["uploaded_retail"] = (
        uploaded_df.copy()
    )
    snapshot_date = (
        uploaded_df["InvoiceDate"].max()
        + pd.Timedelta(days=1)
    )

    uploaded_customer_features = (
        uploaded_df
        .groupby("Customer ID")
        .agg(
            {
                "InvoiceDate":
                lambda x: (
                    snapshot_date - x.max()
                ).days,

                "Invoice": "nunique",

                "total_price": "sum"
            }
        )
        .reset_index()
    )

    uploaded_customer_features.columns = [
        "Customer ID",
        "recency_days",
        "frequency",
        "monetary"
    ]

    # Save uploaded features for dashboard
    st.session_state["uploaded_customer_features"] = (
        uploaded_customer_features.copy()
    )

  

# ------------------------------
# LOAD DATA (ZIP SAFE)
# ------------------------------
@st.cache_data(show_spinner="Loading datasets...")
def load_data():
    raw_csv = DATA_DIR / "raw" / "OnlineRetail_clean.csv"
    feat_path = DATA_DIR / "features" / "customer_features.csv"
    cluster_path = DATA_DIR / "features" / "customer_clusters.csv"

    # ---- Safety checks ----
    for path in [raw_csv, feat_path, cluster_path]:
        if not path.exists():
            st.error(f"❌ Missing file: {path}")
            st.stop()

    # ---- Load data ----
    retail = pd.read_csv(raw_csv)
    customer_features = pd.read_csv(feat_path)
    customer_clusters = pd.read_csv(cluster_path)

    # ---- Feature engineering ----
    retail["InvoiceDate"] = pd.to_datetime(retail["InvoiceDate"])
    retail["InvoiceMonth"] = retail["InvoiceDate"].dt.to_period("M").astype(str)

    return retail, customer_features, customer_clusters

# ------------------------------
# LOAD MODELS
# ------------------------------
@st.cache_resource(show_spinner="Loading models...")
def load_models():
    model_path = MODEL_DIR / "predictor_rf.joblib"

    if not model_path.exists():
        st.error("❌ Model file missing")
        st.stop()

    rf_model = joblib.load(model_path)

    return rf_model

#
# ------------------------------
# INITIAL LOAD
# ------------------------------
retail, customer_features, customer_clusters = load_data()
rf_model = load_models()

kmeans = joblib.load(
    MODEL_DIR / "kmeans_segmentation.joblib"
)


kmeans_scaler = joblib.load(
    MODEL_DIR / "kmeans_scaler.joblib"
)




if "uploaded_retail" in st.session_state:
    retail = st.session_state["uploaded_retail"]

if "uploaded_customer_features" in st.session_state:

    customer_features = (
        st.session_state[
            "uploaded_customer_features"
        ]
    )
  #------------------------------
# PREDICT UPLOADED DATA
# ------------------------------
if uploaded_customer_features is not None:

    try:

        uploaded_customer_features[
            "prediction"
        ] = rf_model.predict(
            uploaded_customer_features[
                [
                    "recency_days",
                    "frequency",
                    "monetary"
                ]
            ]
        )
        st.session_state["uploaded_customer_features"] = (
    uploaded_customer_features.copy()
)

        st.session_state[
    "uploaded_predictions"
] = uploaded_customer_features.copy()



    except Exception as e:

        st.error(
            "Prediction failed on uploaded dataset."
        )

        st.exception(e)


# ------------------------------
# SIDEBAR NAVIGATION
# ------------------------------
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "Overview",
        "Customer Segmentation",
        "RFM Analysis",
        "Customer Prediction",
        
    ],
)

# ------------------------------
# GLOBAL FILTERS
# ------------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("🌍 Global Filters")

countries = ["All"] + sorted(retail["Country"].dropna().unique())
selected_country = st.sidebar.selectbox("Country", countries)

if selected_country != "All":
    retail_filtered = retail[retail["Country"] == selected_country]
else:
    retail_filtered = retail.copy()

# ------------------------------
# PAGE ROUTING
# ------------------------------
if page == "Overview":
    show_overview(retail_filtered, customer_features)

elif page == "Customer Segmentation":

    show_customer_segmentation(
        customer_features,
        customer_clusters
    )
elif page == "RFM Analysis":
    show_rfm(customer_features, retail_filtered)
elif page == "Customer Prediction":
    show_prediction(rf_model)