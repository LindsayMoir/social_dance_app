import pandas as pd
import predibase as pb
import logging
import yaml
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Set up logging
with open("config/config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Set up logging
logging.basicConfig(
    filename=config['logging']['log_file'],
    filemode='w',
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logging.info("Standard configuration loaded successfully.")

# Load the dataset
file_path = config["input"]["predibase"]
df = pd.read_csv(file_path)

# Initialize Predibase client with API key from environment variable
api_key = os.getenv("PREDIBASE_API_KEY")
client = pb.Client(api_key=api_key)
logging.info("Client initialized successfully.")

# Upload dataset to Predibase
dataset = client.create_dataset(name="dance_event_duplicates", file_path=file_path)

# Define model configuration
config = {
    "input_features": [
        {"name": "Group_ID", "type": "category"},  # Helps model compare within groups
        {"name": "event_id", "type": "text"},  
        {"name": "source", "type": "text"},
        {"name": "dance_style", "type": "text"},
        {"name": "url", "type": "text"},
        {"name": "event_type", "type": "text"},
        {"name": "event_name", "type": "text"},
        {"name": "day_of_week", "type": "category"},
        {"name": "start_date", "type": "date"},
        {"name": "end_date", "type": "date"},
        {"name": "start_time", "type": "time"},
        {"name": "end_time", "type": "time"},
        {"name": "price", "type": "text"},
        {"name": "location", "type": "text"},
        {"name": "address_id", "type": "number"},
        {"name": "description", "type": "text"},
        {"name": "time_stamp", "type": "text"},
    ],
    "output_features": [
        {"name": "Label", "type": "number"}
    ],
    "model_type": "llm",  # Large Language Model
    "pretrained_model_name_or_path": "mistral-7b"  # Adjust if needed
}

# Create experiment & train
experiment = client.create_experiment(
    name="duplicate_detection",
    dataset_id=dataset.id,
    config=config
)
logging.info("Experiment created successfully.")

# Start training
experiment.start()

# Monitor training status
print("Training Status:", experiment.status())

# Load the predictions from Predibase
predictions = experiment.get_predictions()

# Select only the required columns
output_df = predictions[["event_id", "Group_ID", "Label"]]

# Save the filtered results to a new CSV
output_df.to_csv("predicted_duplicates.csv", index=False)

# Display first few rows to verify
logging.info(f"Predictions saved successfully. Displaying first few rows: {df.head()}") 