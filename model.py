import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# Load data
def load_data():
    # Replace this with your data loading logic
    data = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [5, 4, 3, 2, 1],
        'target': [1, 2, 3, 4, 5]
    })
    return data

# Train model
def train_model(data):
    X = data[['feature1', 'feature2']]
    y = data['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Save the model
    with open('model.pkl', 'wb') as file:
        pickle.dump(model, file)

if __name__ == '__main__':
    data = load_data()
    train_model(data)
