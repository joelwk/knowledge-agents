from tenacity import retry, wait_random_exponential, stop_after_attempt
import logging
import openai
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError
from typing import Optional, List, Dict, Any, Union, Literal
from enum import Enum
import yaml

# Configuration Models
class ModelProvider(str, Enum):
    """Supported model providers."""
    OPENAI = "openai"
    GROK = "grok"
    VENICE = "venice"

class ModelOperation(str, Enum):
    """Types of model operations."""
    EMBEDDING = "embedding"
    CHUNK_GENERATION = "chunk_generation"
    SUMMARIZATION = "summarization"

class ModelConfig(BaseModel):
    # OpenAI Configuration
    openai_api_key: str
    openai_embedding_model: str
    openai_completion_model: str

    # Grok Configuration
    grok_api_key: Optional[str] = None
    grok_embedding_model: Optional[str] = None
    grok_completion_model: Optional[str] = None

    # Venice Configuration
    venice_api_key: Optional[str] = None
    venice_summary_model: Optional[str] = None  # Specific model for summaries
    venice_chunk_model: Optional[str] = None    # Specific model for chunks

    # Default providers for different operations
    default_embedding_provider: ModelProvider = ModelProvider.OPENAI
    default_chunk_provider: ModelProvider = ModelProvider.OPENAI
    default_summary_provider: ModelProvider = ModelProvider.OPENAI

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
    embedding: Union[List[float], List[List[float]]]
    model: str
    usage: Dict[str, int]

def load_prompts(prompt_path: str = "./prompt.yaml") -> Dict[str, Any]:
    """Load prompts from YAML file."""
    try:
        with open(prompt_path, 'r') as file:
            prompts = yaml.safe_load(file)
        return prompts
    except Exception as e:
        logging.error(f"Error loading prompts: {e}")
        raise

def load_config(config_path: str = "./config.ini"):
    logging.basicConfig(level=logging.ERROR)
    from configparser import ConfigParser
    config_parser = ConfigParser()
    config_parser.read(config_path)
    try:
        model_config = ModelConfig(
            openai_embedding_model=config_parser["models"]["OPENAI_EMBEDDING_MODEL"],
            openai_completion_model=config_parser["models"]["OPENAI_MODEL"],
            grok_completion_model=config_parser["models"].get("GROK_MODEL"),
            grok_embedding_model=config_parser["models"].get("GROK_EMBEDDING_MODEL"),
            venice_summary_model=config_parser["models"].get("VENICE_MODEL_SUMMARY"),
            venice_chunk_model=config_parser["models"].get("VENICE_MODEL_CHUNK"),
            openai_api_key=config_parser["keys"]["OPENAI_API_KEY"],
            grok_api_key=config_parser["keys"].get("GROK_API_KEY"),
            venice_api_key=config_parser["keys"].get("VENICE_API_KEY"),
            default_embedding_provider=config_parser["models"].get("DEFAULT_EMBEDDING_PROVIDER", "openai"),
            default_chunk_provider=config_parser["models"].get("DEFAULT_CHUNK_PROVIDER", "openai"),
            default_summary_provider=config_parser["models"].get("DEFAULT_SUMMARY_PROVIDER", "openai"))

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

class KnowledgeAgent:
    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config
        self.models = self._initialize_clients()
        self.prompts = load_prompts()
        
    def _initialize_clients(self) -> Dict[str, OpenAI]:
        """Initialize model clients dynamically based on provided configuration."""
        clients = {}
        
        # Validate OpenAI configuration
        if not self.model_config.openai_api_key or not self.model_config.openai_api_key.strip():
            logging.warning("OpenAI API key is missing or empty")
        else:
            try:
                openai.api_key = self.model_config.openai_api_key
                clients[ModelProvider.OPENAI] = OpenAI(
                    api_key=self.model_config.openai_api_key,
                    max_retries=5,)
                logging.info("OpenAI client initialized successfully")
            except Exception as e:
                logging.error(f"Failed to initialize OpenAI client: {str(e)}")
        
        # Validate Grok configuration
        if self.model_config.grok_api_key and self.model_config.grok_api_key.strip():
            try:
                clients[ModelProvider.GROK] = OpenAI(
                    api_key=self.model_config.grok_api_key,
                    base_url="https://api.x.ai/v1",
                )
                logging.info("Grok client initialized successfully")
            except Exception as e:
                logging.error(f"Failed to initialize Grok client: {str(e)}")
        # Validate Venice configuration
        if self.model_config.venice_api_key and self.model_config.venice_api_key.strip():
            try:
                clients[ModelProvider.VENICE] = OpenAI(
                    api_key=self.model_config.venice_api_key,
                    base_url="https://api.venice.ai/api/v1",)
                logging.info("Venice client initialized successfully")
            except Exception as e:
                logging.error(f"Failed to initialize Venice client: {str(e)}")
        if not clients:
            raise ValueError("No API providers configured. Please check your API keys in config.ini")
        return clients

    def _get_client(self, provider: ModelProvider) -> OpenAI:
        """Retrieve the appropriate client for a provider."""
        client = self.models.get(provider)
        if not client:
            available_providers = list(self.models.keys())
            if not available_providers:
                raise ValueError("No API providers are configured")
            fallback_provider = available_providers[0]
            logging.warning(f"Provider {provider} not configured, falling back to {fallback_provider}")
            return self.models[fallback_provider]
        return client
    
    def _get_model_name(self, provider: ModelProvider, operation: ModelOperation) -> str:
        """Get the appropriate model name for a provider and operation type."""
        if operation == ModelOperation.EMBEDDING:
            if provider == ModelProvider.OPENAI:
                return self.model_config.openai_embedding_model
            elif provider == ModelProvider.GROK:
                return self.model_config.grok_embedding_model or "grok-v1-embedding"
            else:
                raise ValueError(f"Unsupported embedding provider: {provider}")
        elif operation == ModelOperation.CHUNK_GENERATION:
            if provider == ModelProvider.OPENAI:
                return self.model_config.openai_completion_model
            elif provider == ModelProvider.GROK:
                return self.model_config.grok_completion_model
            elif provider == ModelProvider.VENICE:
                return self.model_config.venice_chunk_model
            else:
                raise ValueError(f"Unsupported completion provider: {provider}")
        elif operation == ModelOperation.SUMMARIZATION:
            if provider == ModelProvider.OPENAI:
                return self.model_config.openai_completion_model
            elif provider == ModelProvider.GROK:
                return self.model_config.grok_completion_model
            elif provider == ModelProvider.VENICE:
                return self.model_config.venice_summary_model
            else:
                raise ValueError(f"Unsupported completion provider: {provider}")
        else:
            raise ValueError(f"Unknown operation: {operation}")

    def _get_default_provider(self, operation: ModelOperation) -> ModelProvider:
        """Get the default provider for a specific operation."""
        provider = None
        if operation == ModelOperation.EMBEDDING:
            provider = self.model_config.default_embedding_provider
            # Only OpenAI and Grok support embeddings
            if provider not in [ModelProvider.OPENAI, ModelProvider.GROK]:
                provider = ModelProvider.OPENAI
        elif operation == ModelOperation.CHUNK_GENERATION:
            provider = self.model_config.default_chunk_provider
        elif operation == ModelOperation.SUMMARIZATION:
            provider = self.model_config.default_summary_provider
        else:
            raise ValueError(f"Unknown operation: {operation}")
        # Validate provider is configured
        if provider not in self.models:
            available_providers = list(self.models.keys())
            if not available_providers:
                raise ValueError("No API providers are configured. Please check your API keys in config.ini")
            provider = available_providers[0]
            logging.warning(f"Default provider not configured, using {provider}")
            
        return provider
    
    def generate_summary(
        self, 
        query: str, 
        results: str, 
        context: Optional[str] = None,
        temporal_context: Optional[Dict[str, str]] = None,
        provider: Optional[ModelProvider] = None
    ) -> str:
        """Generate a summary using the specified provider.
        
        Args:
            query: The original search query
            results: The combined chunk analysis results
            context: Additional analysis context
            temporal_context: Dictionary with start_date and end_date
            provider: The model provider to use
        """
        if provider is None:
            provider = self._get_default_provider(ModelOperation.SUMMARIZATION)
            
        client = self._get_client(provider)
        model = self._get_model_name(provider, ModelOperation.SUMMARIZATION)
        
        # Ensure temporal context is properly formatted
        if temporal_context is None:
            temporal_context = {
                "start_date": "Unknown",
                "end_date": "Unknown"
            }
        
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": self.prompts["system_prompts"]["objective_analysis"]["content"]
                    },
                    {
                        "role": "user",
                        "content": self.prompts["user_prompts"]["summary_generation"]["content"].format(
                            query=query,
                            temporal_context=f"Time Range: {temporal_context['start_date']} to {temporal_context['end_date']}",
                            context=context or "No additional context provided.",
                            results=results,
                            start_date=temporal_context['start_date'],
                            end_date=temporal_context['end_date']
                        )
                    }
                ],
                temperature=0.3,
                presence_penalty=0.2,
                frequency_penalty=0.2
            )
            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"Error generating summary with {provider}: {e}")
            if provider != ModelProvider.OPENAI and ModelProvider.OPENAI in self.models:
                logging.warning(f"Falling back to OpenAI for summary generation")
                return self.generate_summary(
                    query, 
                    results, 
                    context=context,
                    temporal_context=temporal_context,
                    provider=ModelProvider.OPENAI
                )
            raise
    
    def generate_chunks(
        self, 
        content: str,
        provider: Optional[ModelProvider] = None
    ) -> Dict[str, str]:
        """Generate chunks using the specified provider."""
        if provider is None:
            provider = self._get_default_provider(ModelOperation.CHUNK_GENERATION)
            
        client = self._get_client(provider)
        model = self._get_model_name(provider, ModelOperation.CHUNK_GENERATION)
        
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": self.prompts["system_prompts"]["generate_chunks"]["content"]
                    },
                    {
                        "role": "user",
                        "content": self.prompts["user_prompts"]["text_chunk_summary"]["content"].format(
                            content=content
                        )
                    }
                ],
                temperature=0.1,
                presence_penalty=0.1,
                frequency_penalty=0.1
            )
                
            result = response.choices[0].message.content
            sections = result.split("<generated_context>")
            if len(sections) > 1:
                analysis = sections[0].strip()
                context = sections[1].strip()
            else:
                analysis = result
                context = "No specific context generated."
                
            return {
                "analysis": analysis,
                "context": context
            }
            
        except Exception as e:
            logging.error(f"Error generating chunks with {provider}: {e}")
            if provider != ModelProvider.OPENAI and ModelProvider.OPENAI in self.models:
                logging.warning(f"Falling back to OpenAI for chunk generation")
                return self.generate_chunks(content, provider=ModelProvider.OPENAI)
            raise

    @retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(2))
    def embedding_request(
        self,
        text: Union[str, List[str]],
        provider: Optional[ModelProvider] = None
    ) -> EmbeddingResponse:
        """Request embeddings from the specified provider."""
        if provider is None:
            provider = self._get_default_provider(ModelOperation.EMBEDDING)
            
        if provider not in [ModelProvider.OPENAI, ModelProvider.GROK]:
            raise ValueError(f"Unsupported embedding provider: {provider}")
            
        client = self._get_client(provider)
        model = self._get_model_name(provider, ModelOperation.EMBEDDING)
        
        try:
            response = client.embeddings.create(
                input=text,
                model=model,
                encoding_format="float",
                dimensions=3072)
            embeddings = [data.embedding for data in response.data]
            return EmbeddingResponse(
                embedding=embeddings[0] if isinstance(text, str) else embeddings,
                model=model,
                usage=response.usage.model_dump()
            )
            
        except Exception as e:
            logging.error(f"Error getting embeddings from {provider}: {str(e)}")
            if provider != ModelProvider.OPENAI and ModelProvider.OPENAI in self.models:
                logging.warning("Falling back to OpenAI embeddings")
                return self.embedding_request(text, provider=ModelProvider.OPENAI)
            raise