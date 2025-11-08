# ML Pipeline Project

A machine learning pipeline for CPU utilization prediction with support for both local development and Docker deployment.

## Project Structure

```
ML-Pipelines/
├── src/
│   ├── api/
│   │   ├── app.py              # FastAPI application for predictions
│   │   └── training_worker.py  # Background worker for model training
│   ├── models/
│   │   └── generated_cpu_data.py  # Synthetic data generation
│   └── utils/                  # Utility functions
├── data/                       # Data storage
│   ├── cpu_features_raw.csv
│   └── cpu_features_scaled.csv
├── artifacts/
│   └── models/                 # Saved model files
└── client.py                   # API client for interacting with the service

```

## Features


- **Automated Training Pipeline**
  - Hyperparameter optimization with Optuna
  - Continuous training capability
  - Progress monitoring

- **API Features**
  - Real-time predictions
  - Model training requests
  - Training job monitoring


## Setup

1. Create and activate a Python virtual environment:
```bash
conda create -n mlpipeline python=3.8
conda activate mlpipeline
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Start Redis server (required for job management):
```bash
# Option 1: Using system service
sudo service redis-server start

# Option 2: Direct command
redis-server
```

## Usage

### Starting the Service

1. Start the API server:
```bash
uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000
```

2. Generate synthetic training data (if needed):
```bash
python src/models/generated_cpu_data.py
```

### Making Predictions

Use the client to make CPU utilization predictions:

```python
import requests

prediction_request = {
    "request_rate": 120,
    "active_users": 60,
    "db_connections": 25,
    "hour_sin": 0.5,
    "hour_cos": 0.866,
    "minute_sin": 0.2588,
    "minute_cos": 0.9659,
    "is_business_hour": 1
}

response = requests.post("http://localhost:8000/predict", json=prediction_request)
print(response.json())
```

### Training New Models

1. Submit a training job:
```python
response = requests.post("http://localhost:8000/train", 
                       json={"n_trials": 20})
job_id = response.json()["job_id"]
```

2. Monitor training progress:
```python
response = requests.get(f"http://localhost:8000/jobs/{job_id}")
print(response.json())
```

## Model Architecture

The CPU utilization prediction model:

- **Algorithm**: XGBoost Regressor
- **Features**:
  - request_rate: Number of incoming requests
  - active_users: Current active user count
  - db_connections: Database connection count
  - hour_sin/cos: Cyclical time features (hour)
  - minute_sin/cos: Cyclical time features (minute)
  - is_business_hour: Business hours indicator

## API Endpoints

- `POST /predict`: Make CPU utilization predictions
- `POST /train`: Start a new training job
- `GET /train/status/{job_id}`: Get training job status
- `POST /train/cancel/{job_id}`: Cancel running job
- `GET /train/jobs`: Get all jobs
- `GET /models`: List all available models
- `GET /models/{version}/metrics`: Get model metrics
- `POST /models/{version}/activate`: Activate a specific model version





# Future Work for Development


## Monitoring

The system provides real-time monitoring of:
- Model performance metrics
- Training job progress
- Prediction latency
- Error rates

Access monitoring data through the `/metrics` endpoint.

- **Real-time CPU Utilization Prediction**
  - Request rate analysis
  - Active user impact
  - Database connection monitoring
  - Time-based patterns
  - Business hours consideration

- **Model Registry**
  - Model versioning and storage
  - Model metrics access
  - Model activation/deactivation

## License

This project is licensed under the MIT License - see the LICENSE file for details.