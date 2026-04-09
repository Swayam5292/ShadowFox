from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# -------------------------------
# Load Model Safely
# -------------------------------
MODEL_PATH = 'model.pkl'

if os.path.exists(MODEL_PATH):
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None
else:
    print(f"Model file {MODEL_PATH} not found.")
    model = None


# -------------------------------
# Home Route
# -------------------------------
@app.route('/')
def home():
    return render_template('index.html')


# -------------------------------
# Prediction Route
# -------------------------------
@app.route('/predict', methods=['POST'])
def predict():
    
    if model is None:
        return render_template(
            'index.html',
            prediction_text="❌ Model not loaded. Check model.pkl file."
        )

    try:
        # -------------------------------
        # Get Inputs
        # -------------------------------
        present_price = float(request.form.get('Present_Price', 0))
        kms_driven = float(request.form.get('Kms_Driven', 0))
        owner = int(request.form.get('Owner', 0))
        car_age = int(request.form.get('Car_Age', 0))

        fuel = request.form.get('Fuel_Type', 'Petrol')
        seller = request.form.get('Seller_Type', 'Dealer')
        transmission = request.form.get('Transmission', 'Manual')

        # -------------------------------
        # Encoding
        # -------------------------------
        fuel_type_diesel = 1 if fuel == "Diesel" else 0
        fuel_type_petrol = 1 if fuel == "Petrol" else 0

        seller_type_individual = 1 if seller == "Individual" else 0
        transmission_manual = 1 if transmission == "Manual" else 0

        # -------------------------------
        # Create DataFrame (IMPORTANT FIX)
        # -------------------------------
        input_data = pd.DataFrame([{
            'Present_Price': present_price,
            'Kms_Driven': kms_driven,
            'Owner': owner,
            'Car_Age': car_age,
            'Fuel_Type_Diesel': fuel_type_diesel,
            'Fuel_Type_Petrol': fuel_type_petrol,
            'Seller_Type_Individual': seller_type_individual,
            'Transmission_Manual': transmission_manual
        }])

        # -------------------------------
        # Debug (Optional)
        # -------------------------------
        print("Input Data:\n", input_data)

        # -------------------------------
        # Prediction
        # -------------------------------
        prediction = model.predict(input_data)[0]

        result = max(0, round(prediction, 2))

        return render_template(
            'index.html',
            prediction_text=f"₹ {result} Lakhs",
            prediction_val=result,
            inputs=request.form
        )

    except Exception as e:
        print("ERROR:", e)
        return render_template(
            'index.html',
            prediction_text=f"❌ Error: {str(e)}",
            inputs=request.form
        )


# -------------------------------
# Run App
# -------------------------------
if __name__ == "__main__":
    app.run(debug=True)