import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_boston

# Load the dataset
def load_data():
    boston = load_boston()
    df = pd.DataFrame(boston.data, columns=boston.feature_names)
    df['PRICE'] = boston.target
    return df

# Train the model
def train_model():
    df = load_data()

    # Features and target
    X = df.drop('PRICE', axis=1)
    y = df['PRICE']

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2025)

    # Train a Random Forest Regressor
    model = RandomForestRegressor(random_state=2024)
    model.fit(X_train, y_train)

    # Evaluate
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    print(f"Model trained. RMSE: {rmse:.2f}")

    # Save the model and columns
    return model, X.columns.tolist()
