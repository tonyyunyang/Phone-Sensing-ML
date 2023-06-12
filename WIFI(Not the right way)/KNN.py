import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def load_and_parse_data(file_path):
    # Load raw data
    with open(file_path, 'r') as file:
        raw_data = file.readlines()

    # Parse data
    ssids = []
    bssids = []
    rssis = []
    for line in raw_data:
        parts = line.split(',')
        ssids.append(parts[0].split(':')[1].strip())
        bssids.append(parts[1].split(':')[1].strip())
        rssis.append(int(parts[2].split(':')[1].strip()))

    # Create DataFrame
    data = pd.DataFrame({'SSID': ssids, 'BSSID': bssids, 'RSSI': rssis})
    return data

# Load and parse data
east = load_and_parse_data('EastWestAll(06.09_20.00_added)/east_filtered.csv')
west = load_and_parse_data('EastWestAll(06.09_20.00_added)/west_filtered.csv')

# Add labels
east['location'] = 'east'
west['location'] = 'west'

# Combine data
data = pd.concat([east, west])

# Preprocessing
encoder = OneHotEncoder()
scaler = StandardScaler()

bssid = encoder.fit_transform(data['BSSID'].values.reshape(-1, 1)).toarray()
rssi = scaler.fit_transform(data['RSSI'].values.reshape(-1, 1))

# Combine preprocessed BSSID and RSSI
features = np.hstack([bssid, rssi])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, data['location'], test_size=0.05, random_state=42)

# Training
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Prediction
y_pred = knn.predict(X_test)

# Evaluation
print('Accuracy:', accuracy_score(y_test, y_pred))

