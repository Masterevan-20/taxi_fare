try:
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.stats import zscore
    from math import radians, sin, cos, sqrt, atan2  # Thêm các hàm cần thiết từ math
    print("Đã import thành công tất cả thư viện cần thiết")
except ImportError as e:
    print(f"Lỗi import thư viện: {str(e)}")
    print("Vui lòng cài đặt thư viện còn thiếu bằng pip install")

# Đọc dữ liệu từ file uber.csv
print("=== ĐỌC DỮ LIỆU ===")
df = pd.read_csv(r'd:\predict taxi fares\uber.csv', index_col=0)  # Bỏ qua cột index

# Kiểm tra thông tin dữ liệu ban đầu
print("\n=== KIỂM TRA DỮ LIỆU BAN ĐẦU ===")
print("\nThông tin dữ liệu:")
print(df.info())

print("\nMẫu dữ liệu:")
print(df.head())

print("\nThống kê mô tả ban đầu:")
print(df.describe())

# Kiểm tra và xử lý giá trị null
print("\n=== KIỂM TRA VÀ XỬ LÝ GIÁ TRỊ NULL ===")
print("Số lượng giá trị null trong từng cột:")
null_counts = df.isnull().sum()
print(null_counts)

print("\nPhần trăm giá trị null trong từng cột:")
null_percentages = (df.isnull().sum() / len(df)) * 100
print(null_percentages)

# Kiểm tra và xử lý giá trị 0 trong tọa độ
print("\n=== KIỂM TRA VÀ XỬ LÝ GIÁ TRỊ 0 TRONG TỌA ĐỘ ===")
zero_coords = df[(df['pickup_longitude'] == 0) | 
                (df['pickup_latitude'] == 0) |
                (df['dropoff_longitude'] == 0) |
                (df['dropoff_latitude'] == 0)]
print("Số lượng bản ghi có tọa độ bằng 0:", len(zero_coords))

# Thay thế giá trị 0 trong tọa độ bằng median
coord_columns = ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude']
for col in coord_columns:
    median_value = df[df[col] != 0][col].median()
    df[col] = df[col].replace(0, median_value)

# Xử lý giá trị null
print("\n=== XỬ LÝ GIÁ TRỊ NULL ===")
df.fillna(df.median(numeric_only=True), inplace=True)

print("Kiểm tra giá trị null sau khi xử lý:")
print(df.isnull().sum())

# Chuyển đổi cột datetime
print("\n=== CHUYỂN ĐỔI ĐỊNH DẠNG THỜI GIAN ===")
print("Định dạng datetime ban đầu:")
print(df['pickup_datetime'].head())

df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
print("\nĐịnh dạng datetime sau khi chuyển đổi:")
print(df['pickup_datetime'].head())

# Thêm các đặc trưng thời gian
df['hour'] = df['pickup_datetime'].dt.hour
df['day'] = df['pickup_datetime'].dt.day
df['month'] = df['pickup_datetime'].dt.month
df['year'] = df['pickup_datetime'].dt.year

print("\nCác đặc trưng thời gian đã thêm:")
print(df[['hour', 'day', 'month', 'year']].head())

# Thêm thư viện để tính khoảng cách
def haversine_distance(lat1, lon1, lat2, lon2):
    """Tính khoảng cách giữa hai điểm trên trái đất"""
    R = 6371  # Bán kính trái đất (km)
    
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    distance = R * c
    
    return distance

# Thêm sau phần xử lý null và trước phần xử lý ngoại lai
print("\n=== TIỀN XỬ LÝ DỮ LIỆU NÂNG CAO ===")

# 1. Tính khoảng cách chuyến đi
df['trip_distance'] = df.apply(lambda row: haversine_distance(
    row['pickup_latitude'], row['pickup_longitude'],
    row['dropoff_latitude'], row['dropoff_longitude']
), axis=1)

# 2. Xử lý giá cước không hợp lý
print("\nSố chuyến có giá cước <= 0:", len(df[df['fare_amount'] <= 0]))
df = df[df['fare_amount'] > 0]

# 3. Xử lý passenger_count không hợp lý
print("\nPhân bố passenger_count trước khi xử lý:")
print(df['passenger_count'].value_counts().sort_index())
df = df[df['passenger_count'].between(1, 6)]

# 4. Xử lý chuyến đi có điểm đón/trả trùng nhau
duplicate_locations = df[df.apply(lambda row: 
    row['pickup_latitude'] == row['dropoff_latitude'] and 
    row['pickup_longitude'] == row['dropoff_longitude'], axis=1)]
print("\nSố chuyến có điểm đón/trả trùng nhau:", len(duplicate_locations))
df = df[df.apply(lambda row: 
    not (row['pickup_latitude'] == row['dropoff_latitude'] and 
         row['pickup_longitude'] == row['dropoff_longitude']), axis=1)]

# 5. Lọc các chuyến đi quá xa
print("\nThống kê khoảng cách chuyến đi:")
print(df['trip_distance'].describe())
df = df[df['trip_distance'] <= df['trip_distance'].quantile(0.99)]  # Lọc bỏ 1% chuyến xa nhất

# Xử lý ngoại lai
print("\n=== XỬ LÝ NGOẠI LAI ===")
numeric_cols = ['fare_amount', 'pickup_longitude', 'pickup_latitude', 
                'dropoff_longitude', 'dropoff_latitude', 'passenger_count']

print("Thống kê trước khi xử lý ngoại lai:")
print(df[numeric_cols].describe())

# Đếm số lượng ngoại lai trước khi xử lý
print("\nSố lượng ngoại lai trong từng cột:")
outliers_counts = {}
for col in numeric_cols:
    z_scores = np.abs(zscore(df[col]))
    outliers_count = len(df[z_scores > 3])
    outliers_counts[col] = outliers_count
    print(f"{col}: {outliers_count} ngoại lai")

# Lọc ngoại lai một cách an toàn
df_cleaned = df.copy()
mask = np.full(len(df_cleaned), True)

for col in numeric_cols:
    z_scores = np.abs(zscore(df_cleaned[col]))
    mask = mask & (z_scores <= 3)

df_cleaned = df_cleaned[mask]

print(f"\nSố lượng bản ghi ban đầu: {len(df)}")
print(f"Số lượng bản ghi sau khi xử lý ngoại lai: {len(df_cleaned)}")

# Phân tích và xử lý dữ liệu nâng cao
print("\n=== PHÂN TÍCH VÀ XỬ LÝ DỮ LIỆU NÂNG CAO ===")

# 1. Xử lý outliers theo logic nghiệp vụ
print("1. Xử lý outliers theo logic nghiệp vụ...")

# Lọc giá cước hợp lý (từ 2.5$ đến 100$)
df_cleaned = df_cleaned[(df_cleaned['fare_amount'] >= 2.5) & (df_cleaned['fare_amount'] <= 200)]

# Lọc khoảng cách hợp lý (0.1km đến 100km)
df_cleaned = df_cleaned[(df_cleaned['trip_distance'] >= 0.1) & (df_cleaned['trip_distance'] <= 100)]

# Tính và lọc theo tốc độ trung bình
df_cleaned['avg_speed'] = df_cleaned['trip_distance'] / ((df_cleaned['trip_distance']/30) + 0.1)  # km/h
df_cleaned = df_cleaned[df_cleaned['avg_speed'] <= 80]  # Giới hạn tốc độ tối đa 80km/h

# 2. Tạo features mới
print("\n2. Tạo features mới...")

# Features thời gian
df_cleaned['is_weekend'] = df_cleaned['pickup_datetime'].dt.weekday.isin([5, 6]).astype(int)
df_cleaned['is_rush_hour'] = (
    ((df_cleaned['hour'] >= 7) & (df_cleaned['hour'] <= 10)) | 
    ((df_cleaned['hour'] >= 16) & (df_cleaned['hour'] <= 19))
).astype(int)
df_cleaned['day_of_week'] = df_cleaned['pickup_datetime'].dt.dayofweek

# Features khoảng cách
df_cleaned['manhattan_distance'] = (
    abs(df_cleaned['dropoff_longitude'] - df_cleaned['pickup_longitude']) +
    abs(df_cleaned['dropoff_latitude'] - df_cleaned['pickup_latitude'])
) * 111  # Chuyển đổi sang km

# Features giá/khoảng cách
df_cleaned['fare_per_km'] = df_cleaned['fare_amount'] / df_cleaned['trip_distance']
df_cleaned = df_cleaned[df_cleaned['fare_per_km'] <= 50]  # Giới hạn giá/km hợp lý

# 3. Feature selection dựa trên tương quan
print("\n3. Kiểm tra tương quan giữa các biến...")
numeric_cols = ['fare_amount', 'trip_distance', 'passenger_count', 
                'manhattan_distance', 'fare_per_km', 'avg_speed',
                'hour', 'day', 'month', 'year']
correlation_matrix = df_cleaned[numeric_cols].corr()

# Lọc features có tương quan cao với fare_amount
correlation_with_target = abs(correlation_matrix['fare_amount']).sort_values(ascending=False)
strong_corr_features = correlation_with_target[correlation_with_target > 0.1].index.tolist()
strong_corr_features.remove('fare_amount')

print("\nFeatures có tương quan mạnh với fare_amount:")
print(strong_corr_features)

# 4. Kiểm tra và xử lý đa cộng tuyến
from statsmodels.stats.outliers_influence import variance_inflation_factor

def calculate_vif(df, features):
    vif_data = pd.DataFrame()
    vif_data["Feature"] = features
    vif_data["VIF"] = [variance_inflation_factor(df[features].values, i)
                       for i in range(len(features))]
    return vif_data

print("\n4. Kiểm tra đa cộng tuyến...")
vif_df = calculate_vif(df_cleaned, strong_corr_features)
print("\nVIF của các features:")
print(vif_df.sort_values('VIF', ascending=False))

# Loại bỏ features có VIF > 10
final_features = vif_df[vif_df['VIF'] <= 10]['Feature'].tolist()
print("\nFeatures cuối cùng được chọn:")
print(final_features)

# Điều chỉnh kích thước và thiết lập font cho biểu đồ
plt.rcParams['figure.figsize'] = [15, 10]
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

# Thiết lập lại kích thước biểu đồ cho phù hợp với màn hình
plt.figure(figsize=(25, 20))

# 1. Phân tích giá cước và số hành khách
plt.subplot(2, 3, 1)
sns.histplot(data=df_cleaned, x='fare_amount', bins=50, kde=True)
plt.title('Phân phối Fare Amount')
plt.xlabel('Giá cước (USD)')
plt.ylabel('Số lượng')

plt.subplot(2, 3, 2)
sns.countplot(data=df_cleaned, x='passenger_count')
plt.title('Phân phối Passenger Count')
plt.xlabel('Số hành khách')
plt.ylabel('Số lượng chuyến')

plt.subplot(2, 3, 3)
sns.boxplot(data=df_cleaned, x='passenger_count', y='fare_amount')
plt.title('Fare Amount by Passenger Count')
plt.xlabel('Số hành khách')
plt.ylabel('Giá cước (USD)')

# 2. Phân tích theo thời gian
plt.subplot(2, 3, 4)
hourly_fare = df_cleaned.groupby('hour')['fare_amount'].mean()
sns.lineplot(x=hourly_fare.index, y=hourly_fare.values)
plt.title('Giá trung bình theo giờ')
plt.xlabel('Giờ trong ngày')
plt.ylabel('Giá trung bình (USD)')

plt.subplot(2, 3, 5)
monthly_fare = df_cleaned.groupby('month')['fare_amount'].mean()
sns.lineplot(x=monthly_fare.index, y=monthly_fare.values)
plt.title('Giá trung bình theo tháng')
plt.xlabel('Tháng')
plt.ylabel('Giá trung bình (USD)')

plt.subplot(2, 3, 6)
sns.countplot(data=df_cleaned, x='hour')
plt.title('Số chuyến theo giờ')
plt.xlabel('Giờ trong ngày')
plt.ylabel('Số lượng chuyến')

plt.tight_layout()
plt.show()

# 3. Phân tích vị trí đón/trả khách
plt.figure(figsize=(20, 8))

# Heatmap vị trí đón khách
plt.subplot(1, 2, 1)
plt.hist2d(df_cleaned['pickup_longitude'], df_cleaned['pickup_latitude'], bins=50, cmap='YlOrRd')
plt.colorbar(label='Số lượng chuyến')
plt.title('Heatmap vị trí đón khách')
plt.xlabel('Longitude')
plt.ylabel('Latitude')

# Heatmap vị trí trả khách
plt.subplot(1, 2, 2)
plt.hist2d(df_cleaned['dropoff_longitude'], df_cleaned['dropoff_latitude'], bins=50, cmap='YlOrRd')
plt.colorbar(label='Số lượng chuyến')
plt.title('Heatmap vị trí trả khách')
plt.xlabel('Longitude')
plt.ylabel('Latitude')

plt.tight_layout()
plt.show()

# 4. Ma trận tương quan
plt.figure(figsize=(12, 8))
numeric_cols_corr = numeric_cols + ['hour', 'day', 'month', 'year']
correlation_matrix = df_cleaned[numeric_cols_corr].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Ma trận tương quan')
plt.show()

# 5. Thống kê chi tiết
print("\n=== THỐNG KÊ CHI TIẾT ===")
print("\nThống kê về giá cước:")
print(df_cleaned['fare_amount'].describe())

print("\nPhân phối số hành khách:")
print(df_cleaned['passenger_count'].value_counts().sort_index())

print("\nThống kê theo giờ cao điểm (6-9h và 16-19h):")
peak_hours = df_cleaned[
    ((df_cleaned['hour'] >= 6) & (df_cleaned['hour'] <= 9)) |
    ((df_cleaned['hour'] >= 16) & df_cleaned['hour'] <= 19)
]
print("Giá trung bình giờ cao điểm:", peak_hours['fare_amount'].mean())
print("Số chuyến giờ cao điểm:", len(peak_hours))

# Thống kê sau khi xử lý
print("\n=== THỐNG KÊ SAU KHI XỬ LÝ ===")
print(df_cleaned[numeric_cols].describe())

# Lưu dữ liệu đã xử lý
try:
    # Chọn các cột cần thiết cho mô hình
    final_columns = final_features + ['fare_amount', 'is_weekend', 'is_rush_hour', 
                                    'day_of_week', 'fare_per_km', 'avg_speed']
    
    # Lọc bỏ các dòng có giá trị null
    df_final = df_cleaned[final_columns].dropna()
    
    # Lưu file
    output_path = r'd:\predict taxi fares\uber_cleaned.csv'
    df_final.to_csv(output_path, index=False)
    print(f"\nDữ liệu đã được lưu thành công vào file: {output_path}")
    print(f"Số lượng bản ghi trong file đã xử lý: {len(df_final)}")
    print(f"Số lượng features: {len(final_columns)}")
    print("\nThống kê dữ liệu cuối cùng:")
    print(df_final.describe())
except Exception as e:
    print(f"\nLỗi khi lưu file: {str(e)}")
    print("Vui lòng kiểm tra quyền truy cập thư mục và đường dẫn file")

# Phân tích EDA chi tiết
print("\nPhân tích EDA:")
print("\n1. Phân phối giá cước:")
print(f"- Giá trung bình: ${df_cleaned['fare_amount'].mean():.2f}")
print(f"- Giá cao nhất: ${df_cleaned['fare_amount'].max():.2f}")
print(f"- 90% chuyến đi có giá dưới: ${df_cleaned['fare_amount'].quantile(0.9):.2f}")

print("\n2. Phân tích số hành khách:")
passenger_stats = df_cleaned['passenger_count'].value_counts().sort_index()
print("- Phân bố số hành khách:")
print(passenger_stats)

print("\n3. Phân tích theo thời gian:")
peak_hours_stats = df_cleaned[df_cleaned['hour'].isin([7,8,9,17,18,19])]
print(f"- Số chuyến giờ cao điểm: {len(peak_hours_stats)}")
print(f"- Giá trung bình giờ cao điểm: ${peak_hours_stats['fare_amount'].mean():.2f}")

print("\n4. Phân tích khoảng cách:")
print(df_cleaned['trip_distance'].describe())

# Thêm sau phần tính khoảng cách chuyến đi
print("\n=== PHÂN TÍCH VÀ XỬ LÝ ĐA CỘNG TUYẾN ===")

# 1. Tính ma trận tương quan
correlation_matrix = df[['fare_amount', 'pickup_longitude', 'pickup_latitude', 
                        'dropoff_longitude', 'dropoff_latitude', 'passenger_count',
                        'trip_distance', 'hour', 'day', 'month', 'year']].corr()

# Vẽ heatmap tương quan
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Ma trận tương quan giữa các biến')
plt.tight_layout()
plt.show()

# 2. Tính VIF (Variance Inflation Factor) để kiểm tra đa cộng tuyến
from statsmodels.stats.outliers_influence import variance_inflation_factor

def calculate_vif(df, features):
    vif_data = pd.DataFrame()
    vif_data["Feature"] = features
    vif_data["VIF"] = [variance_inflation_factor(df[features].values, i)
                       for i in range(df[features].shape[1])]
    return vif_data

numeric_features = ['pickup_longitude', 'pickup_latitude', 
                   'dropoff_longitude', 'dropoff_latitude',
                   'passenger_count', 'trip_distance', 
                   'hour', 'day', 'month']

print("\nKiểm tra VIF cho các biến:")
vif_df = calculate_vif(df_cleaned, numeric_features)
print(vif_df.sort_values('VIF', ascending=False))

# 3. Tạo thêm các feature mới có ý nghĩa
print("\n=== TẠO THÊM ĐẶC TRƯNG MỚI ===")

# 3.1. Thêm đặc trưng thời gian
df_cleaned['is_weekend'] = df_cleaned['pickup_datetime'].dt.weekday.isin([5, 6]).astype(int)
df_cleaned['is_rush_hour'] = ((df_cleaned['hour'] >= 7) & (df_cleaned['hour'] <= 9) |
                             (df_cleaned['hour'] >= 16) & (df_cleaned['hour'] <= 19)).astype(int)
df_cleaned['time_of_day'] = pd.cut(df_cleaned['hour'], 
                                  bins=[-1, 6, 12, 18, 24], 
                                  labels=['night', 'morning', 'afternoon', 'evening'])

# 3.2. Tạo đặc trưng khoảng cách Manhattan
df_cleaned['manhattan_distance'] = (abs(df_cleaned['dropoff_longitude'] - df_cleaned['pickup_longitude']) +
                                  abs(df_cleaned['dropoff_latitude'] - df_cleaned['pickup_latitude'])) * 111

# 3.3. Tạo đặc trưng tốc độ trung bình (km/h)
df_cleaned['speed'] = df_cleaned['trip_distance'] / (1 + df_cleaned['trip_distance']/30)  # Giả định tốc độ trung bình 30km/h

# 4. Xử lý outliers dựa trên logic domain
print("\n=== XỬ LÝ OUTLIERS DỰA TRÊN LOGIC DOMAIN ===")

# 4.1. Lọc based on speed
df_cleaned = df_cleaned[df_cleaned['speed'] <= 100]  # Giới hạn tốc độ tối đa

# 4.2. Lọc based on fare/distance ratio
df_cleaned['fare_per_km'] = df_cleaned['fare_amount'] / df_cleaned['trip_distance']
df_cleaned = df_cleaned[df_cleaned['fare_per_km'] <= df_cleaned['fare_per_km'].quantile(0.99)]
df_cleaned = df_cleaned[df_cleaned['fare_per_km'] >= df_cleaned['fare_per_km'].quantile(0.01)]

# 4.3. Lọc based on reasonable geographical bounds cho New York
df_cleaned = df_cleaned[
    (df_cleaned['pickup_longitude'].between(-74.3, -73.7)) &
    (df_cleaned['dropoff_longitude'].between(-74.3, -73.7)) &
    (df_cleaned['pickup_latitude'].between(40.5, 41.0)) &
    (df_cleaned['dropoff_latitude'].between(40.5, 41.0))
]

# 5. Feature selection dựa trên tương quan
print("\n=== FEATURE SELECTION ===")
correlation_with_target = abs(correlation_matrix['fare_amount']).sort_values(ascending=False)
print("\nTương quan với biến mục tiêu (fare_amount):")
print(correlation_with_target)

# Chọn các feature có tương quan > 0.1 với fare_amount
selected_features = correlation_with_target[correlation_with_target > 0.1].index.tolist()
selected_features.remove('fare_amount')  # Loại bỏ target variable
print("\nCác feature được chọn:", selected_features)

# Lưu dữ liệu đã xử lý với các feature mới
print("\n=== LƯU DỮ LIỆU ĐÃ XỬ LÝ ===")
try:
    output_path = r'd:\predict taxi fares\uber_cleaned.csv'
    df_cleaned.to_csv(output_path, index=False)
    print(f"\nĐã lưu dữ liệu đã xử lý vào file: {output_path}")
    print(f"Số lượng bản ghi: {len(df_cleaned)}")
    print(f"Số lượng features: {len(df_cleaned.columns)}")
except Exception as e:
    print(f"\nLỗi khi lưu file: {str(e)}")

# Thêm các feature tương tác
df_cleaned['distance_speed'] = df_cleaned['trip_distance'] * df_cleaned['avg_speed']
df_cleaned['passenger_distance'] = df_cleaned['passenger_count'] * df_cleaned['trip_distance']
df_cleaned['peak_distance'] = df_cleaned['is_rush_hour'] * df_cleaned['trip_distance']

# Thêm feature về khu vực
df_cleaned['is_manhattan'] = (
    (df_cleaned['pickup_longitude'].between(-74.0, -73.93)) &
    (df_cleaned['pickup_latitude'].between(40.7, 40.85))
).astype(int)

# Phân tích đa cộng tuyến cho tất cả các biến số
print("\n=== PHÂN TÍCH ĐA CỘNG TUYẾN CHI TIẾT ===")

# Chọn tất cả các cột số
numeric_columns = df_cleaned.select_dtypes(include=['float64', 'int64']).columns.tolist()

# Tính VIF cho tất cả các biến số
vif_data = pd.DataFrame()
vif_data["Feature"] = numeric_columns
vif_data["VIF"] = [variance_inflation_factor(df_cleaned[numeric_columns].values, i)
                   for i in range(len(numeric_columns))]

# Sắp xếp và hiển thị VIF
print("\nChỉ số VIF cho tất cả các biến:")
print(vif_data.sort_values('VIF', ascending=False))

# Vẽ heatmap tương quan cho tất cả các biến
plt.figure(figsize=(15, 12))
correlation_matrix_all = df_cleaned[numeric_columns].corr()
sns.heatmap(correlation_matrix_all, 
            annot=True,
            cmap='coolwarm',
            fmt='.2f',
            square=True)
plt.title('Ma trận tương quan giữa tất cả các biến')
plt.tight_layout()
plt.savefig('correlation_heatmap_all.png')
plt.show()

# Xác định các cặp biến có tương quan cao
print("\nCác cặp biến có tương quan cao (>0.7 hoặc <-0.7):")
high_corr_pairs = []
for i in range(len(numeric_columns)):
    for j in range(i+1, len(numeric_columns)):
        corr = correlation_matrix_all.iloc[i, j]
        if abs(corr) > 0.7:
            high_corr_pairs.append({
                'Variable 1': numeric_columns[i],
                'Variable 2': numeric_columns[j],
                'Correlation': corr
            })

if high_corr_pairs:
    high_corr_df = pd.DataFrame(high_corr_pairs)
    print(high_corr_df.sort_values('Correlation', ascending=False))
else:
    print("Không tìm thấy cặp biến nào có tương quan cao!")

# Lọc và giữ lại các biến dựa trên domain knowledge
print("\nLựa chọn features dựa trên domain knowledge và phân tích VIF:")
selected_features = [
    'trip_distance',    # Giữ lại vì là yếu tố quan trọng nhất với giá cước
    'avg_speed',        # Giữ lại vì ảnh hưởng đến giá cước (phụ phí giờ cao điểm)
    'passenger_count',  # Giữ lại vì có thể ảnh hưởng đến giá cước
    'hour',            # Giữ lại để nắm bắt pranh giờ cao điểm
    'is_weekend',      # Giữ lại vì giá có thể khác vào cuối tuần
    'is_rush_hour',    # Giữ lại vì ảnh hưởng trực tiếp đến giá
    'is_manhattan'     # Giữ lại vì khu vực có thể ảnh hưởng đến giá
]

print("\nCác features được chọn để sử dụng trong mô hình:")
print(selected_features)

# Lưu danh sách features đã chọn để sử dụng trong mô hình
final_features = selected_features.copy()

# Phân tích đa cộng tuyến và tương quan
print("\n=== PHÂN TÍCH ĐA CỘNG TUYẾN VÀ TƯƠNG QUAN CHI TIẾT ===")

# 1. Transform coordinates into meaningful features
df_cleaned['is_manhattan'] = (
    (df_cleaned['pickup_longitude'].between(-74.0, -73.93)) &
    (df_cleaned['pickup_latitude'].between(40.7, 40.85))
).astype(int)

# Add zone-based features (simplified NYC zones)
df_cleaned['is_airport'] = (
    # JFK Airport area
    ((df_cleaned['pickup_longitude'].between(-73.81, -73.77)) &
     (df_cleaned['pickup_latitude'].between(40.63, 40.66))) |
    # LaGuardia Airport area
    ((df_cleaned['pickup_longitude'].between(-73.88, -73.85)) &
     (df_cleaned['pickup_latitude'].between(40.76, 40.78)))
).astype(int)

# 2. Transform time features
df_cleaned['season'] = pd.cut(df_cleaned['month'], 
                            bins=[0, 3, 6, 9, 12], 
                            labels=['Winter', 'Spring', 'Summer', 'Fall'])
df_cleaned['time_period'] = df_cleaned['year'].astype(str) + '-' + df_cleaned['season']

# 3. Define base features after transformation
base_features = [
    'trip_distance',    # Direct distance
    'passenger_count',  # Number of passengers
    'hour',            # Time of day
    'is_weekend',      # Weekend indicator
    'is_rush_hour',    # Rush hour indicator
    'is_manhattan',    # Manhattan area indicator
    'is_airport',      # Airport area indicator
]

# 4. Calculate VIF with improved formatting
def calculate_vif_with_description(df, features):
    vif_data = pd.DataFrame()
    vif_data["Feature"] = features
    vif_data["VIF"] = [variance_inflation_factor(df[features].values, i)
                       for i in range(len(features))]
    
    # Format VIF values without scientific notation
    vif_data["VIF"] = vif_data["VIF"].apply(lambda x: f"{x:.2f}")
    
    # Add collinearity level description
    def get_collinearity_level(vif_val):
        vif = float(vif_val)
        if vif < 5:
            return "Thấp (Chấp nhận được)"
        elif vif < 10:
            return "Trung bình (Cần xem xét)"
        else:
            return "Cao (Nên loại bỏ)"
    
    vif_data["Mức độ đa cộng tuyến"] = vif_data["VIF"].apply(get_collinearity_level)
    return vif_data

# Calculate and display VIF for base features
print("\nKiểm tra VIF cho các features đã chuyển đổi:")
vif_df = calculate_vif_with_description(df_cleaned, base_features)
print(vif_df.sort_values('VIF', ascending=False))

# 5. Calculate correlation matrix for transformed features
correlation_matrix = df_cleaned[base_features + ['fare_amount']].corr()

# Visualize correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, 
            annot=True,
            cmap='coolwarm',
            fmt='.2f',
            square=True)
plt.title('Ma trận tương quan của các features đã chuyển đổi')
plt.tight_layout()
plt.savefig('correlation_transformed_features.png')
plt.show()

# 6. Select final features
selected_features = [feature for feature in base_features]

print("\nCác features cuối cùng được chọn:")
print(selected_features)

# Prepare final dataset
df_final = df_cleaned[selected_features + ['fare_amount']].copy()

# Display final correlation with target
final_correlation = df_final.corr()['fare_amount'].sort_values(ascending=False)
print("\nTương quan với biến mục tiêu (fare_amount):")
print(final_correlation)