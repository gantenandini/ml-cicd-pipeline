# main.py
import sys
import os

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


import os
from src.preprocessing import load_data, preprocess_data, split_data
from src.model import build_model
from src.evaluation import evaluate_model

def run_pipeline(data_file):
    # Load and preprocess data
    df = load_data(data_file)
    df = preprocess_data(df)
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(df)
    
    # Build and train model
    model = build_model()
    model.fit(X_train, y_train)
    
    # Evaluate model
    accuracy = evaluate_model(model, X_test, y_test)
    print(f"Model Accuracy: {accuracy}")
    
if __name__ == "__main__":
    data_file = os.path.join('data', 'raw', 'dataset.csv')  # Adjust the dataset path
    run_pipeline(data_file)
