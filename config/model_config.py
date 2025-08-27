from dotenv import load_dotenv
import os

load_dotenv()

class ModelConfig:
    def __init__(self):
        # API Provider Configuration (Google only)
        self.api_provider = "google"

        # Google API Configuration
        self.google_config = {
            "api_key": os.getenv("GOOGLE_API_KEY"),
            "llm_model": "gemini-2.0-flash",  # Using stable model
            "embedding_model": "models/embedding-001",  # Google's text embedding model
        }

        # Hugging Face Configuration
        self.huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
        if self.huggingface_token:
            os.environ["HUGGINGFACE_HUB_TOKEN"] = self.huggingface_token

    def validate_credentials(self):
        """Validate Google API credentials"""
        return bool(self.google_config["api_key"])

    def get_current_config(self):
        """Get the Google configuration"""
        return self.google_config
