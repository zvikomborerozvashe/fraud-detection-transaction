# Import the necessary libraries
from joblib import load

# Step 1: Load the models
try:
    credit_model = load('./models/credit_card_model.pkl')  # Replace with the correct path
    upi_model = load('./models/upi_model.pkl')            # Replace with the correct path
    print("Models loaded successfully.")
except Exception as e:
    print(f"Error loading models: {e}")
    exit()

# Step 2: Define test data (adjust as needed based on your model)
credit_test_data = [[0.5, 1.2, 3.4, 0.1]]  # Example credit card transaction data
upi_test_data = [[100, 200, 1500]]         # Example UPI transaction data

# Step 3: Test credit card fraud prediction
try:
    credit_prediction = credit_model.predict(credit_test_data)
    print(f"Credit Fraud Prediction: {credit_prediction[0]}")
except Exception as e:
    print(f"Error predicting with credit model: {e}")

# Step 4: Test UPI fraud prediction
try:
    upi_prediction = upi_model.predict(upi_test_data)
    print(f"UPI Fraud Prediction: {upi_prediction[0]}")
except Exception as e:
    print(f"Error predicting with UPI model: {e}")
