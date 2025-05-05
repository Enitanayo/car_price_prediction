import streamlit as st
import pandas as pd
import os
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# App title
st.set_page_config(page_title="Car Price Prediction", layout="centered")
st.title("🚗 Car Price Prediction")

car_makes = ['Toyota', 'Lexus', 'Mercedes-Benz', 'Honda', 'Hyundai', 'Ford',
       'Nissan', 'Acura', 'Land Rover', 'Peugeot', 'Kia', 'BMW', 'Mazda',
       'Mitsubishi', 'Volkswagen', 'Jeep']

# Input form
with st.form("car_form"):
    fuel_type = st.selectbox(
        "Fuel Type",
        options=["-", "Petrol", "Diesel", "Hybrid", "CNG"],
        index=0
    )

    gear_type = st.selectbox(
        "Gear Type",
        options=["-", "Manual", "Automatic", "CVT", "AMT"],
        index=0
    )
    Selling_Condition = st.selectbox(
        "Selling Condition",
        options=["-", "Imported", "Registered", "Brand new"],
        index=0
    )
    
    Bought_Condition = st.selectbox(
        "Bought Condition",
        options=["-", "Imported", "Registered", "Brand new"],
        index=0
    )

# Add empty default for validation
    make = st.selectbox(
        "Car Make",
        options=["-"] + sorted(car_makes),
        index=0
    )

    year = st.number_input("Year of Manufacture", min_value=1900, max_value=2025, step=1)
    
    condition = st.selectbox(
        "Condition",
        options=["-", "Local Used", "Foreign Used", "Brand New"],
        index=0
    )

    mileage = st.number_input("Mileage", min_value=0)
    engine_size = st.number_input("Engine Size (cc)", min_value=0)

    submitted = st.form_submit_button("Predict")

# Form validation and prediction logic
if submitted:
    if (
        fuel_type.startswith("-") or
        gear_type.startswith("-") or
        condition.startswith("-") or
        make.startswith("-")
    ):
        st.warning("⚠️ Please complete all the fields before submitting.")
    else:
        # Prepare input
        data = CustomData(
            fuel_type=fuel_type,
            gear_type=gear_type,
            Make=make.title(),
            Year_of_manufacture=int(year),
            Condition=condition,
            Mileage=float(mileage),
            Engine_size=float(engine_size),
            Selling_Condition=Selling_Condition,
            Bought_Condition=Bought_Condition
        )

        df = data.get_data_as_data_frame()

        # Run prediction
        pipeline = PredictPipeline()
        results = pipeline.predict(df)
        predicted_price = round(results[0])

        # Display result
        st.success(f"💰 The predicted price is: **₦{predicted_price:,}**")
        st.write("### Input Summary", df)

        # Save result to CSV
        df['Predicted_Price'] = predicted_price
        file_path = "artifacts/predictions.csv"
        if os.path.exists(file_path):
            df.to_csv(file_path, mode='a', header=False, index=False)
        else:
            df.to_csv(file_path, index=False)

        st.info("✅ Your input and prediction have been saved to `predictions.csv`.")
