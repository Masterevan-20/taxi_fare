# Import các thư viện cần thiết
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Đọc dữ liệu đã được tiền xử lý
print("=== ĐỌC DỮ LIỆU ĐÃ XỬ LÝ ===")
df = pd.read_csv(r'd:\predict taxi fares\uber_cleaned.csv')
print("Số lượng bản ghi:", len(df))

# Chuẩn bị features và target
print("\n=== CHUẨN BỊ DỮ LIỆU ===")
# Chọn các features quan trọng
features = ['pickup_longitude', 'pickup_latitude', 
           'dropoff_longitude', 'dropoff_latitude',
           'passenger_count', 'hour', 'day', 'month',
           'trip_distance']

X = df[features]
y = df['fare_amount']

# Chia dữ liệu thành tập train và test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Kích thước tập train:", X_train.shape)
print("Kích thước tập test:", X_test.shape)

# Xây dựng và huấn luyện mô hình
print("\n=== HUẤN LUYỆN MÔ HÌNH RANDOM FOREST ===")
# Khởi tạo mô hình Random Forest với các tham số
model = RandomForestRegressor(
    n_estimators=100,  # Số cây trong rừng
    max_depth=None,    # Độ sâu tối đa của mỗi cây
    min_samples_split=2,  # Số mẫu tối thiểu để phân tách node
    min_samples_leaf=1,   # Số mẫu tối thiểu ở node lá
    random_state=42       # Seed để tái tạo kết quả
)

print("Bắt đầu huấn luyện mô hình...")
model.fit(X_train, y_train)
print("Hoàn thành huấn luyện mô hình")

# Dự đoán trên tập test
print("\nThực hiện dự đoán trên tập test...")
y_pred = model.predict(X_test)

# Tính các chỉ số đánh giá
print("\n=== ĐÁNH GIÁ MÔ HÌNH ===")
# 1. MSE (Mean Squared Error)
mse = mean_squared_error(y_test, y_pred)
print(f"MSE: {mse:.4f}")

# 2. RMSE (Root Mean Squared Error)
rmse = np.sqrt(mse)
print(f"RMSE: {rmse:.4f}")

# 3. R² Score
r2 = r2_score(y_test, y_pred)
print(f"R² Score: {r2:.4f}")

# 4. MAPE (Mean Absolute Percentage Error)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
print(f"MAPE: {mape:.4f}%")

# Visualize kết quả
print("\n=== VISUALIZE KẾT QUẢ ===")
plt.figure(figsize=(15, 5))

# 1. Actual vs Predicted
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Giá thực tế')
plt.ylabel('Giá dự đoán')
plt.title('So sánh giá thực tế và dự đoán')

# 2. Residuals Plot
plt.subplot(1, 2, 2)
residuals = y_test - y_pred
plt.scatter(y_pred, residuals, alpha=0.5)
plt.xlabel('Giá dự đoán')
plt.ylabel('Residuals')
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Biểu đồ Residuals')

plt.tight_layout()
plt.show()

# Phân tích độ quan trọng của features
print("\n=== ĐỘ QUAN TRỌNG CỦA CÁC FEATURES ===")
importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': model.feature_importances_
})
print("\nĐộ quan trọng của các features:")
print(importance_df.sort_values(by='Importance', ascending=False))

# Visualize feature importance
plt.figure(figsize=(10, 6))
importance_df.sort_values(by='Importance', ascending=True).plot(
    kind='barh', x='Feature', y='Importance'
)
plt.title('Độ quan trọng của các Features trong Random Forest')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.show()

# Lưu mô hình (tuỳ chọn)
import joblib
joblib.dump(model, r'd:\predict taxi fares\linear_regression_model.pkl')
print("\nĐã lưu mô hình vào file 'linear_regression_model.pkl'")
