from flask import Flask, render_template, request, redirect, url_for, flash
import joblib
import logging
from model import preprocess_user_input

app = Flask(__name__)
app.secret_key = 'your-very-secret-key'  # Needed for flash messages

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model once at startup
try:
    model = joblib.load('trained_model.pkl')
    logger.info('Model loaded successfully.')
except Exception as e:
    logger.error(f'Error loading model: {e}')
    model = None


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        flash('Model is not available. Please try again later.')
        return redirect(url_for('home'))
    
    try:
        user_data = {key: request.form[key] for key in request.form}

        # Type conversions for numerical fields
        numeric_fields = [
            'Age', 'Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed',
            'NumCreditLines', 'InterestRate', 'LoanTerm', 'DTIRatio'
        ]
        for field in numeric_fields:
            user_data[field] = float(user_data[field])

        # Preprocess
        X_user = preprocess_user_input(user_data)
        prediction = model.predict(X_user)[0]
        probability = round(model.predict_proba(X_user)[0][1] * 100, 2)

        result = 'High Risk' if prediction == 1 else 'Low Risk'

        return render_template('results.html', result=result, probability=probability)

    except Exception as e:
        logger.exception('Error in prediction:')
        flash('Error processing your input. Please check your entries.')
        return redirect(url_for('home'))


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
