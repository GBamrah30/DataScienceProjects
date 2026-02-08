import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import joblib
from pathlib import Path

st.set_page_config(page_title = "Kickstarter Dashboard", layout="wide")

BASE_DIR = Path(__file__).resolve().parent.parent

dashboard_data = pd.read_csv(BASE_DIR/"data"/"dashboard_data.csv")
feature_importance = pd.read_csv(BASE_DIR/"data"/"feature_importance.csv")

model = joblib.load(BASE_DIR/"models"/"kickstarter_rf_pipeline.joblib")

st.title("Kickstarter Success Dashboard")

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Overview", "Feature Importance", "Prediction"]
)

if page == "Overview":
    st.header("Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Total Projects", len(dashboard_data))
        
    with col2:
        success_rate = dashboard_data["success"].mean()
        st.metric("Overall Success Rate", f"{success_rate:.2%}")
        
    st.subheader("Sample of Data")
    st.dataframe(dashboard_data.head(50))
    
elif page == "Feature Importance":
    st.header("Most Important Features")
    
    top_features = feature_importance.head(15)
    
    fig = px.bar(
        top_features,
        x = "importance",
        y = "feature",
        orientation="h",
        title = "Top 15 Features Influence Success"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
elif page == "Prediction":
    st.header("Predict Kickstarter Success")

    st.write("Enter details about your Kickstarter project below to get a predicted probability of success.")

    col1, col2 = st.columns(2)
    
    with col1:
        usd_goal_log = st.number_input("Goal (log)", min_value=0.0, value=4.0, step=0.1)
        duration_days = st.number_input("Duration (days)", min_value=1, max_value=60, value=30)
        launch_month = st.selectbox("Launch Month", list(range(1,13)), index=5)
    
    with col2:
        launch_day = st.selectbox("Launch Day", list(range(1,32)), index=14)
        main_category = st.selectbox("Main Category", 
                                     ["Music","Technology","Theater","Fashion","Food","Comics","Crafts",
                                      "Dance","Publishing","Journalism"])
        country = st.selectbox("Country", ["US","GB","CA","AU","DE","FR","Other"])

    if st.button("Predict"):
        input_df = pd.DataFrame({
            "usd_goal_log": [usd_goal_log],
            "duration_days": [duration_days],
            "launch_month": [launch_month],
            "launch_day": [launch_day],
            "main_category": [main_category],
            "country": [country]
        })

        input_df = input_df[["country","main_category","duration_days","usd_goal_log","launch_month","launch_day"]]

        prob = model.predict_proba(input_df)[0][1]

        st.success(f"Predicted Probability of Success: {prob:.2%}")
