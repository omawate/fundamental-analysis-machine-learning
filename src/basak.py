import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from talib import RSI, STOCH
from talib import MACD
from talib import OBV
from talib import WILLR

def exponential_smoothing(series, alpha=0.095):
    """
    Apply exponential smoothing to a time series.
    
    Parameters:
    series (list or numpy array): The time series data.
    alpha (float): The smoothing factor.
    
    Returns:
    list: The smoothed time series.
    """
    result = [series[0]]  # first value is the same as series
    for n in range(1, len(series)):
        result.append(alpha * series[n] + (1 - alpha) * result[n - 1])
    return result

def get_technical_indicators(df):
    """
    Calculate technical indicators for the stock data.
    
    Parameters:
    df (pandas DataFrame): The stock data.
    
    Returns:
    pandas DataFrame: The stock data with technical indicators.
    """
    # Calculate RSI
    df['RSI'] = RSI(df['Close'], timeperiod=14)
    
    # Calculate Stochastic Oscillator
    df['STOCH_K'], df['STOCH_D'] = STOCH(
        df['Close'], df['High'], df['Low'])
    
    # Calculate Williams %R
    df['WilliamsR'] = WILLR(
        df['Close'], df['High'], df['Low'], timeperiod=14)
    
    # Calculate MACD
    df['MACD'], df['MACDSIGNAL'], df['MACDHIST'] = MACD(df['Close'], slowperiod=26, fastperiod=12, signalperiod=9)
    
    # Calculate Price Rate of Change
    df['PriceRateOfChange'] = df['Close'].pct_change(periods=14)
    
    # Calculate On Balance Volume
    df['OBV'] = OBV(df['Close'], df['Volume'])
    
    df = df.dropna()  # Drop rows with any NaN values
    return df

def create_feature_matrix(df, prediction_horizon=60):
    """
    Create the feature matrix and labels from the stock data.
    
    Parameters:
    df (pandas DataFrame): The stock data with technical indicators.
    prediction_horizon (int): The number of days into the future to predict.
    
    Returns:
    tuple: Feature matrix (X) and labels (y).
    """
    # Define the features to be used
    features = ['RSI', 'STOCH_K', 'STOCH_D', 'WilliamsR', 'MACD', 'MACDSIGNAL', 'MACDHIST', 'PriceRateOfChange', 'OBV']
    
    # Feature matrix
    X = df[features]
    
    # Labels: +1 for price increase, 0 for price decrease
    y = df['Close'].shift(-prediction_horizon) >= df['Close']
    #y = y.replace(0, -1)  # replace 0 with -1 for classification purpose
    
    df = pd.concat([X,y], axis = 1)
    df = df.dropna()
    y = df.iloc[:,-1]  # Drop NaN values in labels
    X = df.iloc[:,:-1]  # Drop the last rows from feature matrix to match label length
    return X, y

def train_xgboost(X_train, y_train, X_test, y_test):
    """
    Train the XGBoost model and evaluate its accuracy.
    
    Parameters:
    X_train (pandas DataFrame): The training feature matrix.
    y_train (pandas Series): The training labels.
    X_test (pandas DataFrame): The testing feature matrix.
    y_test (pandas Series): The testing labels.
    
    Returns:
    xgb.XGBClassifier: The trained XGBoost model.
    """
    # Initialize the XGBoost classifier
    model = xgb.XGBClassifier(objective='binary:logistic', n_estimators=100)
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")

    #Calculate precision
    precision = precision_score(y_test, y_pred)
    print(f"Precision: {precision:.2f}")

    #Calculate recall
    recall = recall_score(y_test, y_pred)
    print(f"Recall: {recall:.2f}")
    
    return model

if __name__ == "__main__":
    # Load the stock data (replace 'stock_data.csv' with your actual file)
    df = pd.read_csv('stock_data.csv')
    
    # Apply exponential smoothing to the closing prices
    df['Close'] = exponential_smoothing(df['Close'])
    
    # Extract technical indicators
    df = get_technical_indicators(df)
    
    # Create feature matrix and labels
    X, y = create_feature_matrix(df, prediction_horizon=60)
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the XGBoost model
    model = train_xgboost(X_train, y_train, X_test, y_test)
    
    # Save the trained model
    model.save_model("xgboost_stock_model.json")
