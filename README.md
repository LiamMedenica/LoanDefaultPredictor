Loan Default Prediction Web App

This project is a Flask web application that predicts loan default risk using a Random Forest model trained on a Kaggle loan default dataset. Users can input loan application details through a web form and receive a prediction (Low Risk or High Risk) along with the probability of default.

Project Structure

loan_default_prediction/
├── app.py                   # Main Flask app
├── model.py                 # Data preprocessing and model training
├── templates/
│   ├── index.html           # Home page with input form
│   ├── result.html          # Prediction result page
├── static/
│   ├── css/
│   │   └── style.css       # Styling
│   ├── images/
│   │   └── feature_importance.png  # Feature importance plot
├── data/
│   └── loan_data.csv       # Kaggle dataset (not included, download separately)
├── requirements.txt         # Dependencies
└── README.md               # This file

Setup Instructions

Clone the Repository:
git clone https://github.com/LiamMedenica/LoanDefaultPredictor
cd loan_default_prediction

Create a Virtual Environment:

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

Install Dependencies:
pip install -r requirements.txt

Download Dataset:
https://www.kaggle.com/datasets/nikhil1e9/loan-default
Place loan_data.csv in the data/ folder.

Train the Model:
python model.py
This generates trained_model.pkl and static/images/feature_importance.png.

Run the Flask App:
python app.py
Open http://127.0.0.1:5000 in your browser.

Usage
On the home page, fill in the loan application details (numerical and categorical features).
Click "Predict" to see the default risk and probability.
View the feature importance plot to understand key factors.

Dependencies
Listed in requirements.txt. Main libraries:
Flask: Web framework
Pandas, Scikit-learn: Data processing and modeling
Matplotlib, Seaborn: Visualizations

Notes
Ensure loan_data.csv matches the expected columns (listed in the project description).
The model assumes categorical values match the options in the form (e.g., Education: High School, Bachelor's, etc.).
To retrain the model, rerun model.py after updating the dataset.
