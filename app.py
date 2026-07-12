import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model


st.set_page_config(
    page_title="Hyperspectral Beverage Classification",
    page_icon="🥤",
    layout="wide"
)

# ------------------------------------------------
# Load Model and Preprocessing Objects
# ------------------------------------------------
model = load_model("models/mlp_model.keras")

scaler = joblib.load("models/scaler.pkl")
selector = joblib.load("models/anova_selector.pkl")

# ------------------------------------------------
# Beverage Names
# ------------------------------------------------
class_names = [
    "Papaya",
    "Coffee",
    "Pomegranate",
    "Orange",
    "Tea",
    "Wine",
    "Whisky",
    "Rum",
    "Brandy"
]


st.sidebar.title("📘 Project Information")

st.sidebar.markdown("""
### Project
Hyperspectral Beverage Classification

### Model
MLP Neural Network

### Best Accuracy
**92.12%**

### Input Features
204 Spectral Values

### Beverage Classes
9
""")

st.title("🥤 Hyperspectral Beverage Classification")

st.write("""
Upload an **Excel (.xlsx)** or **CSV (.csv)** file containing **204 hyperspectral spectral values**.

The trained MLP model predicts the beverage class.
""")


uploaded_file = st.file_uploader(
    "Upload Excel or CSV File",
    type=["xlsx", "csv"]
)


if uploaded_file is not None:

    
    if uploaded_file.name.endswith(".csv"):
        data = pd.read_csv(uploaded_file, header=None)
    else:
        data = pd.read_excel(uploaded_file, header=None)

    st.success("✅ File uploaded successfully!")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Rows", data.shape[0])

    with col2:
        st.metric("Columns", data.shape[1])

    st.divider()

    st.subheader("Dataset Preview")

    st.table(data.head())

    

    X = data.values

    X_scaled = scaler.transform(X)

    X_selected = selector.transform(X_scaled)

    probabilities = model.predict(X_selected, verbose=0)

    
    top3 = np.argsort(probabilities, axis=1)[:, -3:][:, ::-1]

    results = []

    for i in range(len(data)):

        first = top3[i][0]
        second = top3[i][1]
        third = top3[i][2]

        results.append({

            "Sample": i + 1,

            "1st Prediction": class_names[first],
            "Confidence (%)":
                round(probabilities[i][first] * 100, 2),

            "2nd Prediction": class_names[second],
            "Confidence 2 (%)":
                round(probabilities[i][second] * 100, 2),

            "3rd Prediction": class_names[third],
            "Confidence 3 (%)":
                round(probabilities[i][third] * 100, 2)

        })

    results_df = pd.DataFrame(results)

    st.divider()

    st.subheader("🏆 Top 3 Predictions")

    st.table(results_df)

    st.success("✅ Prediction completed successfully!")

    csv = results_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📥 Download Prediction Results",
        data=csv,
        file_name="prediction_results.csv",
        mime="text/csv"
    )