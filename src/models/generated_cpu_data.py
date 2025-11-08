import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler



def generate_synthetic_cpu_data(n_samples=1000, scale_data=True):
    # Generate synthetic data for CPU utilization prediction
    np.random.seed(42)

    # Generate basic features with clear patterns
    request_rate = np.random.poisson(100, n_samples)  # Average 100 requests per minute
    active_users = np.random.poisson(50, n_samples)   # Average 50 active users
    db_connections = np.clip(np.random.poisson(20, n_samples), 0, 50)  # Max 50 DB connections

    # Time features
    timestamps = [datetime.now() - timedelta(minutes=x) for x in range(n_samples)]
    hour_of_day = np.array([t.hour for t in timestamps])
    minute_of_hour = np.array([t.minute for t in timestamps])
    is_business_hour = np.where((hour_of_day >= 9) & (hour_of_day <= 17), 1, 0)

    # Create cyclical time features
    hour_sin = np.sin(2 * np.pi * hour_of_day / 24)
    hour_cos = np.cos(2 * np.pi * hour_of_day / 24)
    minute_sin = np.sin(2 * np.pi * minute_of_hour / 60)
    minute_cos = np.cos(2 * np.pi * minute_of_hour / 60)

    # Generate CPU utilization with clear, simple patterns
    base_cpu = 15.0  # Base CPU usage
    request_impact = 0.15 * request_rate  # Each request adds 0.15% CPU
    user_impact = 0.2 * active_users      # Each user adds 0.2% CPU
    db_impact = 0.3 * db_connections      # Each connection adds 0.3% CPU
    time_impact = 5.0 * hour_sin + 2.0 * minute_sin  # Time-based variation
    business_impact = 10.0 * is_business_hour  # Higher during business hours

    # Calculate final CPU utilization
    cpu_utilization = (
        base_cpu +
        request_impact +
        user_impact +
        db_impact +
        time_impact +
        business_impact +
        np.random.normal(0, 1, n_samples)  # Minimal random noise
    )

    # Ensure values are between 0 and 100
    cpu_utilization = np.clip(cpu_utilization, 0, 100)

    # Create feature matrix
    X = pd.DataFrame({
        'request_rate': request_rate,
        'active_users': active_users,
        'db_connections': db_connections,
        'hour_sin': hour_sin,
        'hour_cos': hour_cos,
        'minute_sin': minute_sin,
        'minute_cos': minute_cos,
        'is_business_hour': is_business_hour
    })

    if scale_data:
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
        return X_scaled, cpu_utilization
    else:
        raw_df = X.copy()
        raw_df['cpu_utilization'] = cpu_utilization
        return raw_df

if __name__ == "__main__":
    import os
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data'))
    os.makedirs(data_dir, exist_ok=True)
    
    raw_df = generate_synthetic_cpu_data(n_samples=1000, scale_data=False)
    raw_df.to_csv(os.path.join(data_dir, 'cpu_raw_data.csv'), index=False)

    # Save raw features and target  
    # scaled_X, cpu_utilization = generate_synthetic_cpu_data(n_samples=1000, scale_data=True)
    # scaled_data = scaled_X.copy()
    # scaled_data['cpu_utilization'] = cpu_utilization
    # scaled_data.to_csv(os.path.join(data_dir, 'cpu_scaled_data.csv'), index=False)  
    
