import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('seaborn')
sns.set_palette("husl")

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate the distance between two points on earth using Haversine formula"""
    R = 3959.87433  # Earth's radius in miles
    
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    mi = R * c
    return mi

def preprocess_data(df):
    """Preprocess the data according to the report specifications"""
    print("Starting data preprocessing...")
    
    # Convert pickup_datetime to datetime
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
    
    # Extract time features
    df['hour'] = df['pickup_datetime'].dt.hour
    df['day'] = df['pickup_datetime'].dt.day
    df['month'] = df['pickup_datetime'].dt.month
    df['year'] = df['pickup_datetime'].dt.year
    df['day_of_week'] = df['pickup_datetime'].dt.dayofweek
    
    # Calculate distance
    df['distance'] = haversine_distance(
        df['pickup_latitude'], 
        df['pickup_longitude'],
        df['dropoff_latitude'],
        df['dropoff_longitude']
    )
    
    # Remove invalid records based on report criteria
    df = df[
        (df['fare_amount'] > 0) & 
        (df['fare_amount'] <= 100) &
        (df['passenger_count'] > 0) &
        (df['passenger_count'] <= 6) &
        (df['distance'] <= 31) &
        (df['pickup_longitude'].between(-74.03, -73.77)) &
        (df['pickup_latitude'].between(40.63, 40.85)) &
        (df['dropoff_longitude'].between(-74.03, -73.77)) &
        (df['dropoff_latitude'].between(40.63, 40.85))
    ]
    
    print(f"Shape after preprocessing: {df.shape}")
    return df

def create_features(df):
    """Create feature matrix X and target variable y"""
    features = ['distance', 'passenger_count', 'hour', 'day', 
                'month', 'year', 'day_of_week']
    X = df[features]
    y = df['fare_amount']
    return X, y

def evaluate_model(y_true, y_pred, model_name):
    """Calculate and return evaluation metrics"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    
    return {
        'Model': model_name,
        'RMSE': round(rmse, 2),
        'MSE': round(mse, 2),
        'MAPE': round(mape, 1),
        'R2': round(r2, 2)
    }

def plot_actual_vs_predicted(y_true, y_pred, model_name):
    """Create scatter plot of actual vs predicted values"""
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Actual Fare')
    plt.ylabel('Predicted Fare')
    plt.title(f'Actual vs Predicted Fare Amount - {model_name}')
    plt.tight_layout()
    plt.savefig(f'actual_vs_predicted_{model_name.lower().replace(" ", "_")}.png')
    plt.close()

def main():
    # Load data
    print("Loading data...")
    df = pd.read_csv('uber.csv')
    print(f"Initial shape: {df.shape}")
    
    # Preprocess data
    df = preprocess_data(df)
    
    # Create features
    X, y = create_features(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize models
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=1.0),
        'Decision Tree': DecisionTreeRegressor(max_depth=10),
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=6),
        'XGBoost': xgb.XGBRegressor(n_estimators=100, max_depth=6),
        'LightGBM': lgb.LGBMRegressor(n_estimators=100, max_depth=6)
    }
    
    # Train and evaluate models
    results = []
    print("\nTraining and evaluating models...")
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        # Evaluate model
        metrics = evaluate_model(y_test, y_pred, name)
        results.append(metrics)
        
        # Plot actual vs predicted
        plot_actual_vs_predicted(y_test, y_pred, name)
    
    # Create results DataFrame and save
    results_df = pd.DataFrame(results)
    results_df.to_csv('model_results.csv', index=False)
    print("\nResults saved to model_results.csv")
    
    # Plot feature importance for the best model (XGBoost)
    xgb_model = models['XGBoost']
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': xgb_model.feature_importances_
    })
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance)
    plt.title('Feature Importance (XGBoost)')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()

if __name__ == "__main__":
    main() 