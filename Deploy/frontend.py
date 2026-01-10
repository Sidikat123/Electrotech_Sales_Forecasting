import streamlit as st
import pandas as pd
import joblib
import requests
from datetime import datetime
import plotly.express as px

st.set_page_config(page_title="Sales Forecasting", layout="wide")

# App Title 
st.markdown("<h1 style='text-align: center; color: #004466;'> Electrotech Sales Forecasting App</h1>", unsafe_allow_html=True)

# Sidebar Form 
st.sidebar.header("📋 Forecast Parameters")

with st.sidebar.form("input_form"):
    category = st.selectbox("Select Product Category", ['Accessories', 'Laptop', 'Smartphone', 'Tablet'])
    start_date = st.date_input("Start Date", datetime(2026, 1, 1))
    end_date = st.date_input("End Date", datetime(2026, 6, 1))
    price = st.number_input("Average Price (₦)", min_value=10.0, value=100.0, step=1.0)
    season = st.selectbox("Season", ['Winter', 'Spring', 'Summer', 'Fall'])
    submit = st.form_submit_button(" Predict")

# Forecast Logic 
if submit:
    if start_date >= end_date:
        st.error("End date must be after start date.")
    else:
        payload = {
            "category": category,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "price": price,
            "season": season
        }

        try:
            response = requests.post("https://kl8fjd4z-8001.uks1.devtunnels.ms/predict", json=payload)
            response.raise_for_status()
            results = response.json()

            # Convert to DataFrame
            forecast_df = pd.DataFrame(results)
            forecast_df['ds'] = pd.to_datetime(forecast_df['ds'])
            forecast_df.rename(columns={"ds": "Date", "prediction": "Predicted Sales Volume"}, inplace=True)
            forecast_df['Predicted Sales Volume'] = forecast_df['Predicted Sales Volume'].round().astype(int)

            st.success("Forecast complete!")

            # Metrics
            col1, col2 = st.columns([1, 3])
            with col1:
                st.metric("📦 Months Forecasted", len(forecast_df))
                st.metric("🔼 Max Forecast", f"{forecast_df['Predicted Sales Volume'].max():,} units")
                st.metric("🔽 Min Forecast", f"{forecast_df['Predicted Sales Volume'].min():,} units")

            with col2:
                st.dataframe(forecast_df.style.format({"Predicted Sales Volume": "{:,} units"}))

            # Plot
            fig = px.line(
                forecast_df,
                x="Date",
                y="Predicted Sales Volume",
                markers=True,
                title=f"{category} Sales Volume Forecast ({season})",
                labels={"Predicted Sales Volume": "Sales Volume (units)", "Date": "Date"}
            )

            fig.update_traces(line=dict(width=3))
            fig.update_layout(
                title_x=0.5,
                yaxis_tickformat=",",
                yaxis_title="Sales Volume (units)",
                xaxis_title="Date",
                template="simple_white"
            )

            st.plotly_chart(fig, use_container_width=True)

        except requests.exceptions.RequestException as e:
            st.error(f"Something went wrong: {e}")
