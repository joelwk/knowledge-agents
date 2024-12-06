import csv
import glob
import configparser
from tenacity import retry, wait_random_exponential, stop_after_attempt
import logging
import openai
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError
from typing import Optional, List, Dict, Any, Union
from enum import Enum

# Configuration Models
class EmbeddingProvider(str, Enum):
    """Supported embedding providers."""
    OPENAI = "openai"
    GROK = "grok"

class ModelConfig(BaseModel):
    openai_embedding_model: str
    openai_model: str
    grok_model: Optional[str] = None
    grok_embedding_model: Optional[str] = None
    venice_model: Optional[str] = None
    openai_api_key: str
    grok_api_key: Optional[str] = None
    venice_api_key: Optional[str] = None
    default_embedding_provider: EmbeddingProvider = EmbeddingProvider.OPENAI

    class Config:
        use_enum_values = True

class AppConfig(BaseModel):
    root_path: str
    all_data: str
    all_data_stratified_path: str
    knowledge_base: str
    sample_size: int = Field(gt=0)
    filter_date: Optional[str]

class EmbeddingResponse(BaseModel):
    """Standardized embedding response across providers."""
    embedding: Union[List[float], List[List[float]]]  # Can be single embedding or list of embeddings
    model: str
    usage: Dict[str, int]

# Configuration Loader
def load_config(config_path: str = "./config_template.ini"):
    logging.basicConfig(level=logging.ERROR)
    from configparser import ConfigParser
    config_parser = ConfigParser()
    config_parser.read(config_path)
    try:
        model_config = ModelConfig(
            openai_embedding_model=config_parser["models"]["OPENAI_EMBEDDING_MODEL"],
            openai_model=config_parser["models"]["OPENAI_MODEL"],
            grok_model=config_parser["models"].get("GROK_MODEL"),
            grok_embedding_model=config_parser["models"].get("GROK_EMBEDDING_MODEL"),
            venice_model=config_parser["models"].get("VENICE_MODEL"),
            openai_api_key=config_parser["keys"]["OPENAI_API_KEY"],
            grok_api_key=config_parser["keys"].get("GROK_API_KEY"),
            venice_api_key=config_parser["keys"].get("VENICE_API_KEY"),
            default_embedding_provider=config_parser["models"].get("DEFAULT_EMBEDDING_PROVIDER", "openai"))

        app_config = AppConfig(
            root_path=config_parser["data"]["ROOT_PATH"],
            all_data=config_parser["data"]["ALL_DATA"],
            all_data_stratified_path=config_parser["data"]["ALL_DATA_STRATIFIED_PATH"],
            knowledge_base=config_parser["data"]["KNOWLEDGE_BASE"],
            sample_size=int(config_parser["configuration_params"]["SAMPLE_SIZE"]),
            filter_date=config_parser["configuration_params"].get("FILTER_DATE"),)
        return model_config, app_config
    except ValidationError as e:
        logging.error(f"Configuration Error: {e}")
        raise

# Knowledge Agent
class KnowledgeAgent:
    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config
        self.models = self._initialize_clients()
        
    def _initialize_clients(self):
        """Initialize model clients dynamically based on provided configuration."""
        clients = {}
        if self.model_config.openai_api_key:
            openai.api_key = self.model_config.openai_api_key
            clients["openai"] = OpenAI(api_key=self.model_config.openai_api_key)
        
        if self.model_config.grok_api_key:
            clients["grok"] = OpenAI(
                api_key=self.model_config.grok_api_key,
                base_url="https://api.x.ai/v1"
            )
        
        if self.model_config.venice_api_key:
            clients["venice"] = OpenAI(
                api_key=self.model_config.venice_api_key,
                base_url="https://api.venice.ai/api/v1"
            )
        return clients

    def _get_client(self, model_name: str):
        """Retrieve the appropriate client."""
        client = self.models.get(model_name)
        if not client:
            raise ValueError(f"Model {model_name} is not configured.")
        return client
    
    def _get_model_name(self, model_name: str, for_embeddings: bool = False) -> str:
        """Retrieve the appropriate model name based on the client and usage."""
        if for_embeddings:
            if model_name == "openai":
                return self.model_config.openai_embedding_model
            elif model_name == "grok":
                return self.model_config.grok_embedding_model or "grok-v1-embedding"
            else:
                raise ValueError(f"Unsupported embedding model: {model_name}")
        else:
            if model_name == "openai":
                return self.model_config.openai_model
            elif model_name == "grok":
                return self.model_config.grok_model
            elif model_name == "venice":
                return self.model_config.venice_model
            else:
                raise ValueError(f"Unsupported model name: {model_name}")
    
    def generate_summary(self, query: str, results: str, model_name: str = "grok") -> str:
        """Generate a summary using the specified model."""
        client = self._get_client(model_name)
        model = self._get_model_name(model_name)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": f"""Objective Analysis Directive

                    Extract and summarize the provided key points and insights with utmost objectivity, adhering to the following principles:

                    1. **Neutrality**: Abstain from injecting personal biases, emotions, or subjective worldviews.
                    2. **Factuality**: Focus exclusively on verifiable information, avoiding speculative or unsubstantiated claims.
                    3. **Impartiality**: Refrain from promoting or criticizing any particular ideology, entity, or individual.
                    4. **Clarity**: Present the summary in a concise, easily understandable manner, avoiding ambiguity.
                    5. **Source Agnosticism**: Do not attribute information to specific sources, unless explicitly required by the query.
                    **Summary Requirements**
                    **Key Points**: Enumerate the main findings, stripped of interpretative language.
                    **Summary**:
                        + **Main Event**: Describe the central occurrence, devoid of emotional tone.
                        + **Key Insights**: Highlight the most critical, fact-based understandings.
                        + **Actionable Items**: List concrete, unbiased steps, if applicable.
                        + **Speculated Outcomes**: Clearly label and present any forward-looking statements, ensuring they are grounded in the provided information.""",},
                {
                    "role": "user",
                    "content": f"""Write a summary from this collection of key points extracted from the social network dialog.
                        The summary should highlight the main event, key insights, actionable items, and speculated outcomes. If if the prompt requests financial information, provide the assets to be considered.
                        User query: {query}
                        Key points:\n{results}\nSummary:\n""",
                },],
            temperature=0,
        )
        return response.choices[0].message.content
    
    def generate_chunks(self, content: str, template_prompt: str, model_name: str = "grok") -> str:
        """Generate chunks using a template prompt."""
        client = self._get_client(model_name)
        model = self._get_model_name(model_name)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": template_prompt + content}
            ],
            temperature=0)
        return response.choices[0].message.content

    @retry(wait=wait_random_exponential(min=1, max=40), stop=stop_after_attempt(3))
    def embedding_request(
        self,
        text: Union[str, List[str]],
        provider: Optional[str] = None
    ) -> EmbeddingResponse:
        """Request embeddings from the specified provider.
        
        Args:
            text: Text or list of texts to embed
            provider: Provider to use (openai or grok). If None, uses default from config.
            
        Returns:
            Standardized embedding response with single embedding or list of embeddings
        """
        if provider is None:
            provider = self.model_config.default_embedding_provider
            
        if provider not in [EmbeddingProvider.OPENAI, EmbeddingProvider.GROK]:
            raise ValueError(f"Unsupported embedding provider: {provider}")
            
        client = self._get_client(provider)
        model = self._get_model_name(provider, for_embeddings=True)
        
        try:
            response = client.embeddings.create(
                input=text,
                model=model
            )
            
            # Standardize the response
            embeddings = [data.embedding for data in response.data]
            
            return EmbeddingResponse(
                embedding=embeddings[0] if isinstance(text, str) else embeddings,
                model=model,
                usage=response.usage.model_dump()
            )
            
        except Exception as e:
            logging.error(f"Error getting embeddings from {provider}: {e}")
            # If primary provider fails and we have a fallback, try it
            if provider == EmbeddingProvider.GROK and "openai" in self.models:
                logging.warning("Falling back to OpenAI embeddings")
                return self.embedding_request(text, provider=EmbeddingProvider.OPENAI)
            raise