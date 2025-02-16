import yaml
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI, Cohere, Anthropic, MistralAI

class LLMClient:
    def __init__(self, config_path="config.yaml"):
        self.config = self.load_config(config_path)
        self.llm = self.initialize_llm()

    def load_config(self, config_path):
        with open(config_path, "r") as file:
            return yaml.safe_load(file)

    def initialize_llm(self):
        """Dynamically selects the correct LLM provider based on config.yaml."""
        provider = self.config["llm"]["provider"]
        model = self.config["llm"]["model"]
        api_key = self.config["llm"]["api_key"]

        # Dictionary mapping provider names to LangChain LLM classes
        llm_providers = {
            "openai": lambda: ChatOpenAI(model_name=model, openai_api_key=api_key),
            "cohere": lambda: Cohere(model=model, cohere_api_key=api_key),
            "anthropic": lambda: Anthropic(model=model, anthropic_api_key=api_key),
            "mistral": lambda: MistralAI(model=model, mistral_api_key=api_key),
        }

        # Select the correct provider or raise an error if unsupported
        if provider not in llm_providers:
            raise ValueError(f"Unsupported provider: {provider}")

        return llm_providers[provider]()

    def generate_response(self, prompt):
        """Generates a response using the selected LLM provider."""
        return self.llm.predict(prompt)

