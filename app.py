from flask import Flask, render_template, request
import numpy as np
import logging
import os
from joblib import load

# Initialize Flask app
app = Flask(__name__)

# Logging configuration
logging.basicConfig(level=logging.DEBUG)

# Load models with error handling
try:
    credit_model = load('C:/Users/JIN/Documents/projects/code/Fraud_Detection_Transaction-master/Fraud_Detection_Transaction-master/models/credit_card_model.pkl')
    logging.info("Credit fraud model loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load 'credit_card_model.pkl': {e}")
    credit_model = None

try:
    upi_model = load('C:/Users/JIN/Documents/projects/code/Fraud_Detection_Transaction-master/Fraud_Detection_Transaction-master/models/upi_model.pkl')
    logging.info("UPI fraud model loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load 'upi_model.pkl': {e}")
    upi_model = None

# Routes
@app.route('/')
def home():
    """Home page."""
    return render_template('index.html')

@app.route('/credit_fraud', methods=['GET', 'POST'])
def credit_fraud():
    """Handle credit fraud detection."""
    if request.method == 'POST':
        # Get features input from form
        features_input = request.form.get('features', '').strip()
        if not features_input:
            return render_template(
                'credit.html', 
                error="Features cannot be empty.", 
                features_input=features_input
            )
        
        try:
            # Convert features to numpy array
            features = [float(x) for x in features_input.split(',')]
            features = np.array(features).reshape(1, -1)

            # Check if model is available
            if credit_model is None:
                raise RuntimeError("Credit fraud model is unavailable.")

            # Predict using the model
            prediction = credit_model.predict(features)
            result = (
                "Fraudulent transaction detected" 
                if prediction[0] == 1 else 
                "Not a fraudulent transaction"
            )
            return render_template('result.html', title="Credit Fraud Result", result=result)
        except ValueError:
            logging.error("Invalid input: Non-numeric values provided.")
            return render_template(
                'credit.html', 
                error="Invalid input. Please enter numeric values separated by commas.", 
                features_input=features_input
            )
        except Exception as e:
            logging.error(f"Error during credit fraud prediction: {e}")
            return render_template(
                'credit.html', 
                error="An unexpected error occurred during prediction.", 
                features_input=features_input
            )

    return render_template('credit.html')

@app.route('/upi_fraud', methods=['GET', 'POST'])
def upi_fraud():
    """Handle UPI fraud detection."""
    if request.method == 'POST':
        try:
            # Get input data from form
            withdrawal = float(request.form.get('Withdrawal', '0').strip())
            deposit = float(request.form.get('Deposit', '0').strip())
            balance = float(request.form.get('Balance', '0').strip())

            # Prepare data for prediction
            input_data = np.array([withdrawal, deposit, balance]).reshape(1, -1)

            # Check if model is available
            if upi_model is None:
                raise RuntimeError("UPI fraud model is unavailable.")

            # Predict using the model
            prediction = upi_model.predict(input_data)
            result = (
                "Fraudulent UPI transaction detected" 
                if prediction[0] == 1 else 
                "Not a fraudulent UPI transaction"
            )
            return render_template('result.html', title="UPI Fraud Result", result=result)
        except ValueError:
            logging.error("Invalid input: Non-numeric values provided.")
            return render_template(
                'upi.html', 
                error="Invalid input. Please enter numeric values.", 
                withdrawal=request.form.get('Withdrawal', ''),
                deposit=request.form.get('Deposit', ''),
                balance=request.form.get('Balance', '')
            )
        except Exception as e:
            logging.error(f"Error during UPI fraud prediction: {e}")
            return render_template(
                'upi.html', 
                error="An unexpected error occurred during prediction.", 
                withdrawal=request.form.get('Withdrawal', ''),
                deposit=request.form.get('Deposit', ''),
                balance=request.form.get('Balance', '')
            )

    return render_template('upi.html')

# Run Flask app
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    debug_mode = os.environ.get("DEBUG", "False").lower() == "true"
    app.run(host='0.0.0.0', port=port, debug=debug_mode)
