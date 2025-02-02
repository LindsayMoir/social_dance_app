from fastapi import FastAPI
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os

# Load environment variables
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://social_dance_db_user:your_password@your_render_host/social_dance_db")

# Database connection
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Initialize FastAPI app
app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "FastAPI Backend is Running!"}

@app.get("/events")
def get_events():
    with engine.connect() as conn:
        result = conn.execute("SELECT * FROM events LIMIT 10")
        return [dict(row) for row in result]
