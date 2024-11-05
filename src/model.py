import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def generate_data(num_samples):
    np.random.seed(42)
    data = pd.DataFrame({
        'temperature': np.random.uniform(20, 120, num_samples),
        'vibration': np.random.uniform(0.5, 1.5, num_samples),
        'pressure': np.random.uniform(100, 150, num_samples),
        'age': np.random.uniform(0, 1000, num_samples),
        'rpm': np.random.uniform(3000, 4000, num_samples),
        'fuel_consumption': np.random.uniform(10, 15, num_samples),
        'oil_temperature': np.random.uniform(90, 110, num_samples),
        'battery_voltage': np.random.uniform(10, 12, num_samples),
        'brake_pad_wear': np.random.uniform(0, 5, num_samples),
        'tire_pressure': np.random.uniform(25, 35, num_samples),
        'ambient_temperature': np.random.uniform(20, 40, num_samples ),
        'road_condition_smooth': np.random.binomial(1, 0.5, num_samples),
        'road_condition_rough': np.random.binomial(1, 0.3, num_samples),
        'road_condition_potholes': np.random.binomial(1, 0.2, num_samples),
        'driving_style_aggressive': np.random.binomial(1, 0.4, num_samples),
        'driving_style_normal': np.random.binomial(1, 0.5, num_samples),
        'driving_style_gentle': np.random.binomial(1, 0.1, num_samples),
        'mileage': np.random.uniform(100000, 200000, num_samples),
        'time_since_last_maintenance': np.random.uniform(100, 500, num_samples),
        'component_age': np.random.uniform(0, 10, num_samples),
        'rpm_to_age': np.random.uniform(500, 1000, num_samples),
        'failure': np.random.binomial(1, 0.1, num_samples)  # 10% failure rate
    })
    return data

def train_model(data):
    X = data.drop('failure', axis=1)
    y = data['failure']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    return model, X_train, y_train

def predict_failure(model, features):
    features = np.array(features).reshape(1, -1)
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]
    return prediction, probability