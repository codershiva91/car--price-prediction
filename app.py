
# Required Libraries
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS, cross_origin
import pickle
import pandas as pd
import numpy as np
import os
# Initialize Flask App
app = Flask(__name__)
cors = CORS(app)

# Load Model & Data
model = pickle.load(open("LinearRegressionModel.pkl", 'rb'))
car = pd.read_csv("Cleaned_Car_data.csv")


# Home Route
@app.route('/')
def index():
    companies = sorted(car['company'].unique())
    car_models = sorted(car['name'].unique())
    years = sorted(car['year'].unique(), reverse=True)  # Sort in descending order
    fuel_types = car['fuel_type'].unique()  # Convert to list

    companies.insert(0, 'Select Company')
    return render_template('index.html', companies=companies, car_models=car_models, years=years, fuel_types=fuel_types)


# ✅ New Route: Fetch Car Models Based on Selected Company
@app.route('/get_car_models', methods=['POST'])
def get_car_models():
    company = request.form.get('company')

    if not company or company == "Select Company":
        return jsonify({"models": []})  # Return empty if no valid company is selected

    models = sorted(car[car['company'] == company]['name'].unique())  # Filter models based on company
    return jsonify({"models": models})

#
# # Prediction Route
# @app.route('/predict', methods=['POST'])
# @cross_origin()
# def predict():
#     try:
#         company = request.form.get('company')
#         car_model = request.form.get('car_model')
#         year = (request.form.get('year'))
#         fuel_type = request.form.get('fuel_type')
#         driven = (request.form.get('kms_driven'))  # Convert to integer
#
#         # Ensure all inputs are present
#         if not all([company, car_model, year, fuel_type, driven]):
#             return "Error: Missing Input Data"
#
#         # Prediction
#         prediction = model.predict(pd.DataFrame(columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'],
#                                                 data=np.array([car_model, company, year, driven, fuel_type]).reshape(1,
#                                                                                                                      5)))
#
#         return str(np.round(prediction[0], 2))
#
#     except Exception as e:
#         return f"Error: {str(e)}"  # Return error message

@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    # Get form data
    company = request.form.get('company')
    car_model = request.form.get('car_model')
    year = request.form.get('year')
    fuel_type = request.form.get('fuel_type')
    driven = request.form.get('kms_driven')  # Ensure this matches input field name

    # Debugging: Print received values
    print(f"Received: company={company}, model={car_model}, year={year}, fuel_type={fuel_type}, driven={driven}")

    # Check for missing values
    if None in [company, car_model, year, fuel_type, driven]:
        return "Error: Missing input values", 400  # Return error message

    try:
        # Convert year and kms_driven to integers
        year = int(year)
        driven = int(driven)

        # Make prediction
        prediction = model.predict(pd.DataFrame(columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'],
                                                data=np.array([car_model, company, year, driven, fuel_type]).reshape(1, 5)))
        return str(np.round(prediction[0], 2))

    except ValueError as e:
        return f"Error: {e}", 400  # Return error message if conversion fails


# Run Server on Port 5001
# if __name__ == "__main__":
#     app.run(port=5001, debug=True)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)

