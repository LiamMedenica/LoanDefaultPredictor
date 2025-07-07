import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load and preprocess dataset
def preprocess_data():
    # Load dataset
    df = pd.read_csv('data/loan_data.csv')
    
    # Drop LoanID
    df = df.drop('LoanID', axis=1)
    
    # Define features
    numerical_features = ['Age', 'Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed', 
                         'NumCreditLines', 'InterestRate', 'LoanTerm', 'DTIRatio']
    categorical_features = ['Education', 'EmploymentType', 'MaritalStatus', 
                           'HasMortgage', 'HasDependents', 'LoanPurpose', 'HasCoSigner']
    
    # Split features and target
    X = df.drop('Default', axis=1)
    y = df['Default']
    
    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
        ])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and train model
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    model.fit(X_train, y_train)
    
    # Save model
    joblib.dump(model, 'trained_model.pkl')
    
    # Generate feature importance plot
    feature_names = (numerical_features + 
                     model.named_steps['preprocessor']
                     .named_transformers_['cat']
                     .get_feature_names_out(categorical_features).tolist())
    importances = model.named_steps['classifier'].feature_importances_
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances, y=feature_names)
    plt.title('Feature Importance in Loan Default Prediction')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig('static/images/feature_importance.png')
    plt.close()
    
    return model

# Preprocess user input to match training data
def preprocess_user_input(user_data):
    # Convert user input to DataFrame
    df_user = pd.DataFrame([user_data])
    
    return df_user

# Run preprocessing and training
if __name__ == '__main__':
    preprocess_data()