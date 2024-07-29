import pandas as pd
import os
import numpy as np
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix



def rmse(x):
    return np.sqrt(np.mean(x**2))

def preprocess(data, counter, original_filename):
    data = data.sort_values('timestamp')
    data = data.reset_index(drop=True)
    data = data.dropna(axis=0)
    print(counter, " Data info \n")
    print("Each sensor has : \n")
    print(data['sensor'].value_counts())
    linear_acceleration_sensors = []
    rotation_sensors = []
    gyroscope_sensors = []

    for index, row in data.iterrows():
        sensor_type = row['sensor']
        if sensor_type == 'Linear Acceleration':
            row['magnitude'] = np.sqrt(row['x']**2 + row['y']**2 + row['z']**2)
            linear_acceleration_sensors.append(row.tolist())  
        elif sensor_type == 'Rotation Vector':
            row['magnitude'] = np.sqrt(row['x']**2 + row['y']**2 + row['z']**2)
            rotation_sensors.append(row.tolist()) 
        elif sensor_type == 'Gyroscope':
            row['magnitude'] = np.sqrt(row['x']**2 + row['y']**2 + row['z']**2)
            gyroscope_sensors.append(row.tolist())

    min_len = min(len(linear_acceleration_sensors), len(rotation_sensors), len(gyroscope_sensors))

    combined_sensors = []
    for i in range(min_len):
        combined_row = linear_acceleration_sensors[i] + rotation_sensors[i] + gyroscope_sensors[i]
        combined_sensors.append(combined_row)
    
    new_columns = ["id_lin_acc","age_lin_acc","sensor_lin_acc","timestamp_lin_acc","userId_lin_acc","x_lin_acc","y_lin_acc","z_lin_acc","magnitude_lin_acc",
                   "id_rotation","age_rotation","sensor_rotation","timestamp_rotation","userId_rotation","x_rotation","y_rotation","z_rotation","magnitude_rotation",
                   "id_gyro","age_gyro","sensor_gyro","timestamp_gyro","userId_gyro","x_gyro","y_gyro","z_gyro","magnitude_gyro"]
    combined_df = pd.DataFrame(combined_sensors, columns=new_columns)
    combined_df = combined_df.drop(columns=['id_lin_acc','sensor_lin_acc','timestamp_lin_acc','id_rotation','age_rotation','sensor_rotation','timestamp_rotation','id_gyro','age_gyro','sensor_gyro','timestamp_gyro'])

    output_filename = f"preprocessed_{original_filename}"
    output_folder = os.path.join(os.getcwd(), "Preprocessed_Data")
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, output_filename)
    combined_df.to_csv(output_path, index=False)
    
    return combined_df

# Preprocessing step
folder_path = os.path.join(os.getcwd(), "Data")
counter = 1
for file_name in os.listdir(folder_path):
    if file_name.endswith('.csv'):
        file_path = os.path.join(folder_path, file_name)
        data = pd.read_csv(file_path)
        counter += 1
        print('Before preprocessing \n')
        print(counter, " csv found \n")
        print(data.head(), "\n")
        print('------------------------------\n')
        data = preprocess(data, counter, file_name)
        print('After preprocessing \n')
        print('-----------------------------\n')
        print(data.head())

# Processing step
def process_file(file_path):
    df = pd.read_csv(file_path)
    features = ['x_lin_acc', 'y_lin_acc', 'z_lin_acc', 'magnitude_lin_acc',
                'x_rotation', 'y_rotation', 'z_rotation', 'magnitude_rotation',
                'x_gyro', 'y_gyro', 'z_gyro', 'magnitude_gyro']

    results = {}
    for feature in features:
        results[f'{feature}_mean'] = df[feature].mean()
        results[f'{feature}_max'] = df[feature].max()
        results[f'{feature}_min'] = df[feature].min()
        results[f'{feature}_rmse'] = rmse(df[feature])
        results[f'{feature}_std'] = df[feature].std()
    
    results['userId'] = df['userId_lin_acc'].iloc[0]
    results['age_lin_acc'] = 1 if (df['age_lin_acc'].iloc[0] >= 0 and df['age_lin_acc'].iloc[0] <= 13) else 0
    
    results['label'] = results['age_lin_acc']
    
    return results

directory = os.path.join(os.getcwd(), 'Preprocessed_Data')

csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]
all_results = []

for csv_file in csv_files:
    file_path = os.path.join(directory, csv_file)
    result = process_file(file_path)
    all_results.append(result)

final_df = pd.DataFrame(all_results)
final_df=final_df.drop(columns=['age_lin_acc'])
print(final_df.columns)
final_df.columns = ['la_x_mean', 'la_y_mean', 'la_z_mean', 'la_x_max', 'la_y_max',
                    'la_z_max', 'la_x_min', 'la_y_min', 'la_z_min', 'la_x_rmse',
                    'la_y_rmse', 'la_z_rmse', 'la_x_std', 'la_y_std', 'la_z_std',
                    'la_mag_mean', 'la_mag_max', 'la_mag_min', 'la_mag_rmse', 'la_mag_std',
                    'ro_x_mean', 'ro_y_mean', 'ro_z_mean', 'ro_x_max', 'ro_y_max',
                    'ro_z_max', 'ro_x_min', 'ro_y_min', 'ro_z_min', 'ro_x_rmse',
                    'ro_y_rmse', 'ro_z_rmse', 'ro_x_std', 'ro_y_std', 'ro_z_std',
                    'ro_mag_mean', 'ro_mag_max', 'ro_mag_min', 'ro_mag_rmse', 'ro_mag_std',
                    'gy_x_mean', 'gy_y_mean', 'gy_z_mean', 'gy_x_max', 'gy_y_max',
                    'gy_z_max', 'gy_x_min', 'gy_y_min', 'gy_z_min', 'gy_x_rmse',
                    'gy_y_rmse', 'gy_z_rmse', 'gy_x_std', 'gy_y_std', 'gy_z_std',
                    'gy_mag_mean', 'gy_mag_max', 'gy_mag_min', 'gy_mag_rmse', 'gy_mag_std','id', 
                    'label']

print(final_df)
selected_features = [
    'ro_y_max', 'la_z_mean', 'ro_y_mean', 'la_z_rmse', 'ro_x_rmse',
    'la_mag_mean', 'la_x_min', 'ro_y_min', 'ro_y_rmse', 'la_x_mean',
    'ro_x_mean', 'ro_z_max', 'la_mag_std', 'ro_x_min', 'gy_y_std',
    'la_z_max', 'la_mag_rmse', 'la_y_rmse', 'gy_y_rmse', 'ro_mag_max',
    'ro_z_mean', 'ro_mag_rmse','label'
]

ml_data = final_df[selected_features]
ml_data.to_csv('ml_data.csv', index=False)
