import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from lightgbm import LGBMClassifier
import joblib
import os

def prepare_features(df):
    """Prepare features for model training"""
    # Create a copy to avoid modifying the original dataframe
    df_processed = df.copy()
    
    # Label encode categorical variables
    le_geography = LabelEncoder()
    le_gender = LabelEncoder()
    
    df_processed['Geography'] = le_geography.fit_transform(df_processed['Geography'])
    df_processed['Gender'] = le_gender.fit_transform(df_processed['Gender'])
    
    # Save the label encoders
    joblib.dump(le_geography, 'models/le_geography.joblib')
    joblib.dump(le_gender, 'models/le_gender.joblib')
    
    # Select features for model
    features = ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance',
               'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary',
               'Complain', 'Satisfaction Score']
    
    return df_processed[features], df_processed['Exited']

def train_model():
    """Train and save the churn prediction model"""
    # Load data
    df = pd.read_csv('Customer-Churn-Records.csv')
    
    # Prepare features
    X, y = prepare_features(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = LGBMClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)
    
    # Save model and scaler
    joblib.dump(model, 'models/churn_model.joblib')
    joblib.dump(scaler, 'models/scaler.joblib')
    
    # Save feature names
    feature_names = X.columns.tolist()
    joblib.dump(feature_names, 'models/feature_names.joblib')
    
    return model, scaler, feature_names, X_test, y_test

def predict_churn(data):
    """Make predictions using the trained model"""
    # Load model and preprocessing objects
    model = joblib.load('models/churn_model.joblib')
    scaler = joblib.load('models/scaler.joblib')
    le_geography = joblib.load('models/le_geography.joblib')
    le_gender = joblib.load('models/le_gender.joblib')
    
    # Process input data
    data_processed = data.copy()
    data_processed['Geography'] = le_geography.transform(data_processed['Geography'])
    data_processed['Gender'] = le_gender.transform(data_processed['Gender'])
    
    # Select and order features
    features = ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance',
               'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary',
               'Complain', 'Satisfaction Score']
    X = data_processed[features]
    
    # Scale features
    X_scaled = scaler.transform(X)
    
    # Make prediction
    prob = model.predict_proba(X_scaled)[:, 1]
    return prob

def get_feature_importance(model=None):
    """Get feature importance from the trained model"""
    if model is None:
        model = joblib.load('models/churn_model.joblib')
    feature_names = joblib.load('models/feature_names.joblib')
    
    importance = model.feature_importances_
    feature_imp = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    return feature_imp

if __name__ == '__main__':
    # Create models directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # Train and save model
    model, scaler, feature_names, X_test, y_test = train_model()
    
    # Print model performance metrics
    from sklearn.metrics import classification_report
    y_pred = model.predict(scaler.transform(X_test))
    print("\nModel Performance Report:")
    print(classification_report(y_test, y_pred)) 