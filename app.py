import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pickle

st.title("Concrete Strength Prediction App")

st.sidebar.header("Upload your CSV data")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Data Preview")
    st.dataframe(df.head())

    features = [
        "cement", "slag", "flyash", "water",
        "superplasticizer", "coarseaggregate",
        "fineaggregate", "age"
    ]
    target = "csMPa"

    x = df[features]
    y = df[target]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    model = XGBRegressor(
        n_estimators=100,
        learning_rate=0.01,
        subsample=0.08,
        colsample_bytree=0.8,
        max_depth=8,
        random_state=42
    )
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.write(f"Mean Squared Error: {mse:.2f}")
    st.write(f"R2 Score: {r2:.2f}")

    st.header("Make a Prediction")
    user_data = {}
    for feat in features:
        min_val = float(df[feat].min())
        max_val = float(df[feat].max())
        mean_val = float(df[feat].mean())
        user_data[feat] = st.number_input(feat.capitalize(), float(min_val), float(max_val), float(mean_val))

    if st.button("Predict Concrete Strength"):
        user_df = pd.DataFrame([user_data])
        pred = model.predict(user_df)[0]
        st.success(f"Predicted Concrete Strength (csMPa): {pred:.2f}")

    # Save model if requested
    if st.button("Download Model"):
        pickle_bytes = pickle.dumps(model)
        st.download_button(
            label="Download Model",
            data=pickle_bytes,
            file_name="xgb_concrete_model.pkl"
        )
else:
    st.info("Please upload a CSV file containing columns: cement, slag, flyash, water, superplasticizer, coarseaggregate, fineaggregate, age, csMPa")

