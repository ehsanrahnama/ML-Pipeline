import os
from pathlib import Path

# Determine if running in Docker
IN_DOCKER = os.environ.get('DOCKER_ENV', '').lower() == 'true'

# Base directories
if IN_DOCKER:
    BASE_DIR = Path('/shared_data')
    ARTIFACTS_DIR = BASE_DIR / 'artifacts'
    DATA_DIR = BASE_DIR
    MODEL_DB = os.environ.get('MODEL_DB', 'sqlite:////shared_data/model_registry.db')
    REDIS_HOST = 'redis'
else:
    BASE_DIR = Path(__file__).parent.parent.parent
    ARTIFACTS_DIR = BASE_DIR / 'artifacts'
    DATA_DIR = BASE_DIR / 'data'
    MODEL_DB = os.environ.get('MODEL_DB', f'sqlite:///{BASE_DIR}/data/model_registry.db')
    REDIS_HOST = 'localhost'

# Create directories if they don't exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
(ARTIFACTS_DIR / 'models').mkdir(parents=True, exist_ok=True)

# File paths
CPU_FEATURES_RAW = DATA_DIR / 'cpu_features_raw.csv'
CPU_FEATURES_SCALED = DATA_DIR / 'cpu_features_scaled.csv'
CPU_TARGET = DATA_DIR / 'cpu_target.csv'
INITIAL_MODEL_PATH = ARTIFACTS_DIR / 'models' / 'xgboost_model_init.joblib'

# Redis configuration
REDIS_PORT = 6379
REDIS_DB = 0

# API configuration
API_HOST = "0.0.0.0"
API_PORT = 8000

# Training configuration
DEFAULT_N_TRIALS = 20
MAX_CONCURRENT_JOBS = 2