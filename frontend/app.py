from __future__ import annotations

import os

import requests
import streamlit as st


API_URL = os.getenv("UFO_API_URL", "http://localhost:8000")


st.set_page_config(page_title="UFO Country Predictor", page_icon="UFO", layout="centered")

st.title("UFO Country Predictor")
st.caption("Task 1 local deployment: Streamlit frontend + FastAPI backend")

with st.form("prediction_form"):
    seconds = st.number_input("Duration in seconds", min_value=1.0, max_value=60.0, value=50.0)
    latitude = st.number_input("Latitude", min_value=-90.0, max_value=90.0, value=44.0)
    longitude = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=-12.0)
    submitted = st.form_submit_button("Predict")

if submitted:
    payload = {
        "seconds": seconds,
        "latitude": latitude,
        "longitude": longitude,
    }

    try:
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
        response.raise_for_status()
        result = response.json()
    except requests.RequestException as exc:
        st.error(f"Backend request failed: {exc}")
    else:
        st.metric("Predicted country", result["country"])
        st.write(f"Encoded country class: `{result['country_code']}`")
        if result.get("model_accuracy") is not None:
            st.write(f"Validation accuracy from training: `{result['model_accuracy']:.4f}`")

with st.expander("Backend settings"):
    st.write(f"API URL: `{API_URL}`")
    st.write("Run the backend with: `uvicorn backend.main:app --reload --port 8000`")

