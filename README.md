# Taxi Fare Prediction

This project implements a machine learning pipeline to predict taxi fares based on various features such as pickup/dropoff locations, time, and passenger count.

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Make sure you have the `uber.csv` file in the same directory as the script.

3. Run the script:
```bash
python taxi_fare_prediction.py
```

## Features

The script performs the following tasks:
- Data preprocessing and cleaning
- Feature engineering (distance calculation, time features)
- Training and evaluation of 8 different regression models
- Generation of performance metrics (RMSE, MSE, MAPE, R²)
- Visualization of actual vs predicted values
- Feature importance analysis

## Output

The script generates:
1. `model_results.csv` - Contains performance metrics for all models
2. `actual_vs_predicted_*.png` - Scatter plots for each model
3. `feature_importance.png` - Feature importance plot for XGBoost model

## Models Used

1. Linear Regression
2. Ridge Regression
3. Lasso Regression
4. Decision Tree
5. Random Forest
6. Gradient Boosting
7. XGBoost
8. LightGBM 