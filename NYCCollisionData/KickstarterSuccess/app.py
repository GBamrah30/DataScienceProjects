import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import joblib
from pathlib import Path

# ===================================================
# PAGE CONFIGURATION
# ===================================================
st.set_page_config(
    page_title="Kickstarter Success Dashboard",
    layout="wide"
)

# ===================================================
# BASE DIRECTORY
# ===================================================
BASE_DIR = Path(__file__).resolve().parent

# ===================================================
# CACHED DATA LOADERS
# ===================================================
@st.cache_data
def load_dashboard_data(base_dir):
    df = pd.read_csv(base_dir / "data" / "dashboard_data.csv")
    df.columns = df.columns.str.strip()
    
    # Ensure numeric columns are numeric
    for col in ["usd_goal_log", "duration_days", "launch_day", "launch_month", "success"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

@st.cache_data
def load_feature_importance(base_dir):
    df = pd.read_csv(base_dir / "data" / "feature_importance.csv")
    df.columns = df.columns.str.strip()
    return df

@st.cache_resource
def load_model(base_dir):
    return joblib.load(base_dir / "models" / "kickstarter_rf_pipeline.joblib")

# Load data
dashboard_data = load_dashboard_data(BASE_DIR)
feature_importance = load_feature_importance(BASE_DIR)
model = load_model(BASE_DIR)

# ===================================================
# TITLE + SIDEBAR NAVIGATION
# ===================================================
st.title("Kickstarter Success Dashboard")
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Overview", "Feature Importance", "Prediction"]
)

# ===================================================
# OVERVIEW PAGE
# ===================================================
if page == "Overview":
    st.header("Project Overview")

    # High-Level Metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Projects", f"{len(dashboard_data):,}")
    with col2:
        success_rate = dashboard_data["success"].mean() if "success" in dashboard_data.columns else 0
        st.metric("Overall Success Rate", f"{success_rate:.1%}")

    st.divider()

    # ---------------------------------------------
    # Distribution of Funding Goals (log)
    # ---------------------------------------------
    if "usd_goal_log" in dashboard_data.columns and dashboard_data["usd_goal_log"].notna().any():
        st.subheader("Funding Goal Distribution")
        fig1 = px.histogram(
            dashboard_data.dropna(subset=["usd_goal_log"]),
            x="usd_goal_log",
            nbins=50,
            title="Distribution of Funding Goals (log)",
            labels={"usd_goal_log": "Funding Goal (log)"}
        )
        st.plotly_chart(fig1, use_container_width=True)

    # ---------------------------------------------
    # Distribution of Campaign Duration
    # ---------------------------------------------
    if "duration_days" in dashboard_data.columns and dashboard_data["duration_days"].notna().any():
        st.subheader("Campaign Duration Distribution")
        fig2 = px.histogram(
            dashboard_data.dropna(subset=["duration_days"]),
            x="duration_days",
            nbins=30,
            title="Distribution of Campaign Duration (days)",
            labels={"duration_days": "Campaign Duration (days)"}
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ---------------------------------------------
    # Success Rate by Main Category
    # ---------------------------------------------
    if "main_category" in dashboard_data.columns and "success" in dashboard_data.columns:
        st.subheader("Success Rate by Main Category")
        category_success = dashboard_data.groupby("main_category")["success"].mean().reset_index()
        category_success = category_success.sort_values("success", ascending=False)

        fig3 = px.bar(
            category_success,
            x="main_category",
            y="success",
            title="Success Rate by Main Category",
            text="success",
            labels={"main_category": "Category", "success": "Success Rate"}
        )
        fig3.update_traces(texttemplate="%{text:.1%}", textposition="outside")
        st.plotly_chart(fig3, use_container_width=True)

    st.divider()
    st.subheader("Sample Data")
    st.dataframe(dashboard_data.head(50))

# ===================================================
# FEATURE IMPORTANCE PAGE
# ===================================================
elif page == "Feature Importance":
    st.header("What Drives Kickstarter Success?")
    st.write(
        "These features had the strongest influence on the machine learning model’s predictions. "
        "Higher importance means the model relied more heavily on that feature."
    )

    # Slider lets user control how many features to show
    top_n = st.slider("Number of features to display", 5, 25, 15)
    top_features = feature_importance.sort_values("importance", ascending=True).tail(top_n)

    fig = px.bar(
        top_features,
        x="importance",
        y="feature",
        orientation="h",
        title="Most Influential Features"
    )
    st.plotly_chart(fig, use_container_width=True)

# ===================================================
# PREDICTION PAGE
# ===================================================
elif page == "Prediction":
    st.header("Predict Kickstarter Success")
    st.info("Enter project details below to estimate the probability of success.")

    col1, col2 = st.columns(2)
    with col1:
        # Funding goal in dollars (user-friendly)
        usd_goal = st.number_input("Funding Goal ($)", min_value=1.0, value=10000.0, step=100.0)
        duration_days = st.number_input("Campaign Duration (days)", min_value=1, value=30)  # No max limit
        launch_month = st.selectbox("Launch Month", list(range(1, 13)), index=5)
    with col2:
        launch_day = st.selectbox("Launch Day", list(range(1, 32)), index=14)
        main_category = st.selectbox(
            "Main Category",
            dashboard_data["main_category"].unique().tolist()
        )
        country = st.selectbox(
            "Country",
            dashboard_data["country"].unique().tolist()
        )

    if st.button("Predict"):
        # Convert funding goal to log for model
        usd_goal_log = np.log1p(usd_goal)  # log(1 + value) to handle small amounts

        input_df = pd.DataFrame({
            "usd_goal_log": [usd_goal_log],
            "duration_days": [duration_days],
            "launch_month": [launch_month],
            "launch_day": [launch_day],
            "main_category": [main_category],
            "country": [country]
        })

        input_df = input_df[["country", "main_category", "duration_days", "usd_goal_log", "launch_month", "launch_day"]]

        prob = model.predict_proba(input_df)[0][1]

        fig = px.pie(
            names=["Failure", "Success"],
            values=[1 - prob, prob],
            hole=0.5,
            title="Predicted Outcome Probability"
        )
        st.plotly_chart(fig, use_container_width=True)
        st.success(f"Estimated Probability of Success: {prob:.1%}")
