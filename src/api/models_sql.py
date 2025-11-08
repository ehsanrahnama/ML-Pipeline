import os
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, JSON, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime

from src.config.settings import MODEL_DB, INITIAL_MODEL_PATH
from src.utils import get_logger_by_name

logger = get_logger_by_name('models-sql')

engine = create_engine(MODEL_DB, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class ModelEntry(Base):
    __tablename__ = "models"
    id = Column(Integer, primary_key=True, index=True)
    version = Column(String, unique=True, index=True)
    path = Column(String) 
    params = Column(String)  
    metrics = Column(String)  
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    active = Column(Boolean, default=False)
    status = Column(String, default="trained")  

def init_db():
    Base.metadata.create_all(bind=engine)

def initialize_database():
    db = SessionLocal()
    try:
        # Add any initial setup or seed data if necessary
        existing_model = db.query(ModelEntry).filter(ModelEntry.version == "v1_main").first()
        if not existing_model:
            initial_model = ModelEntry(
                version="v1_main",
                path=str(INITIAL_MODEL_PATH),
                params='{"n_estimators": 1000}',
                metrics='{"rmse":0.0837, "mae": 0.0632, "r2": 0.7143}',
                active=True,
                status="deployed"
            )
            db.add(initial_model)
            db.commit()
            logger.info(f"Initial model 'v1_main' seeded successfully at {INITIAL_MODEL_PATH}")
        else:
            logger.info("Initial model 'v1_main' already exists. Skipping seed.")
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        db.rollback()
        raise
    finally:
        db.close()

if __name__ == "__main__":
    init_db()
    initialize_database()
    logger.info("DB initialized")