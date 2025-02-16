# llmclient.py

from datetime import datetime
import yaml
import os
import logging
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain_community.llms import Cohere, Anthropic
from langchain_mistralai import ChatMistralAI


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("llm_client.log"),  # Log to file
        logging.StreamHandler()  # Log to console
    ]
)
logging.info("Initializing LLM client...")

class LLMClient:
    def __init__(self, config_path="config.yaml"):
        load_dotenv()  # Load environment variables from .env
        self.config = self.load_config(config_path)
        self.llm = self.initialize_llm()

    def load_config(self, config_path):
        """Loads configuration from a YAML file."""
        try:
            with open(config_path, "r") as file:
                config = yaml.safe_load(file)
            logging.info(f"def load_config(): Config loaded successfully from {config_path}")
            return config
        except Exception as e:
            logging.error(f"def load_config*(: Error loading config file: {e}")
            raise

    def initialize_llm(self):
        """Dynamically selects the correct LLM provider based on config.yaml."""
        provider = self.config["llm"]["provider"]
        model = self.config["llm"]["model"]
        logging.info(f"Initializing LLM: Provider={provider}, Model={model}")

        # Load API keys from .env
        api_keys = {
            "openai": os.getenv("OPENAI_API_KEY"),
            "cohere": os.getenv("COHERE_API_KEY"),
            "anthropic": os.getenv("ANTHROPIC_API_KEY"),
            "mistral": os.getenv("MISTRAL_API_KEY"),
        }

        if provider not in api_keys:
            logging.error(f"Unsupported provider: {provider}")
            raise ValueError(f"Unsupported provider: {provider}")

        api_key = api_keys[provider]
        if not api_key:
            logging.error(f"Missing API key for provider: {provider}")
            raise ValueError(f"Missing API key for provider: {provider}")

        llm_providers = {
            "openai": lambda: ChatOpenAI(model_name=model, openai_api_key=api_key),
            "cohere": lambda: Cohere(model=model, cohere_api_key=api_key),
            "anthropic": lambda: Anthropic(model=model, anthropic_api_key=api_key),
            "mistral": lambda: ChatMistralAI(model=model, mistral_api_key=api_key),
        }

        try:
            llm = llm_providers[provider]()
            logging.info(f"def initialize_llm(): LLM initialized successfully: {provider} - {model}")
            return llm
        except Exception as e:
            logging.error(f"def initialize_llm(): Failed to initialize LLM: {e}")
            raise

    def generate_response(self, prompt):
        """Generates a response using the selected LLM provider."""
        logging.info(f"def generate_response(): Generating response for prompt: {prompt[:50]}...")  # Log only first 50 chars

        try:
            response = self.llm.predict(prompt)
            logging.info(f"def initialize_llm(): LLM response received successfully")
            return response
        except Exception as e:
            logging.error(f"def initialize_llm(): Error generating response: {e}")
            return "def initialize_llm(): An error occurred while generating the response."

if __name__ == "__main__":

    start_time = datetime.now()

    client = LLMClient()
    prompt = "What is the meaning of life?"
    response = client.generate_response(prompt)
    print(response)

    end_time = datetime.now()
    elapsed_time = end_time - start_time
    logging.info(f"Elapsed time: {elapsed_time}")
