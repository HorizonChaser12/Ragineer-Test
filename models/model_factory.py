from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import logging

logger = logging.getLogger(__name__)

class ModelFactory:
    @staticmethod
    def create_embedding_model(config, api_provider):
        """Create embedding model - only supports Google Gemini now"""
        try:
            if api_provider == "google":
                return GoogleGenerativeAIEmbeddings(
                    model="models/embedding-001",
                    google_api_key=config["api_key"],
                )
            else:
                logger.error(f"Only 'google' API provider is supported, got: {api_provider}")
                return None
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            return None

    @staticmethod
    def create_llm_model(config, api_provider, temperature=0.7):
        """Create LLM model - only supports Google Gemini now"""
        try:
            if api_provider == "google":
                return GoogleGenerativeAI(
                    model=config["llm_model"],
                    google_api_key=config["api_key"],
                    temperature=temperature,
                )
            else:
                logger.error(f"Only 'google' API provider is supported, got: {api_provider}")
                return None
        except Exception as e:
            logger.error(f"Failed to initialize LLM model: {e}")
            return None
            logger.error(f"Failed to initialize LLM model: {e}")
            return None
