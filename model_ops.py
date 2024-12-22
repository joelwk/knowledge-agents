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
    venice_completion_model: Optional[str] = None

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

# Configuration Loader
def load_config(config_path: str = "./config_template.ini"):
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
            venice_completion_model=config_parser["models"].get("VENICE_MODEL"),
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

# Knowledge Agent
class KnowledgeAgent:
    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config
        self.models = self._initialize_clients()
        self.prompts = load_prompts()
        
    def _initialize_clients(self) -> Dict[str, OpenAI]:
        """Initialize model clients dynamically based on provided configuration."""
        clients = {}
        if self.model_config.openai_api_key:
            openai.api_key = self.model_config.openai_api_key
            clients[ModelProvider.OPENAI] = OpenAI(api_key=self.model_config.openai_api_key)
        
        if self.model_config.grok_api_key:
            clients[ModelProvider.GROK] = OpenAI(
                api_key=self.model_config.grok_api_key,
                base_url="https://api.x.ai/v1"
            )
        
        if self.model_config.venice_api_key:
            clients[ModelProvider.VENICE] = OpenAI(
                api_key=self.model_config.venice_api_key,
                base_url="https://api.venice.ai/api/v1"
            )
        return clients

    def _get_client(self, provider: ModelProvider) -> OpenAI:
        """Retrieve the appropriate client for a provider."""
        client = self.models.get(provider)
        if not client:
            raise ValueError(f"Provider {provider} is not configured.")
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
        else:  # For completion models (chunks and summaries)
            if provider == ModelProvider.OPENAI:
                return self.model_config.openai_completion_model
            elif provider == ModelProvider.GROK:
                return self.model_config.grok_completion_model
            elif provider == ModelProvider.VENICE:
                return self.model_config.venice_completion_model
            else:
                raise ValueError(f"Unsupported completion provider: {provider}")

    def _get_default_provider(self, operation: ModelOperation) -> ModelProvider:
        """Get the default provider for a specific operation."""
        if operation == ModelOperation.EMBEDDING:
            return self.model_config.default_embedding_provider
        elif operation == ModelOperation.CHUNK_GENERATION:
            return self.model_config.default_chunk_provider
        elif operation == ModelOperation.SUMMARIZATION:
            return self.model_config.default_summary_provider
        else:
            raise ValueError(f"Unknown operation: {operation}")
    
    def generate_summary(
        self, 
        query: str, 
        results: str, 
        context: Optional[str] = None,
        provider: Optional[ModelProvider] = None
    ) -> str:
        """Generate a summary using the specified provider."""
        if provider is None:
            provider = self._get_default_provider(ModelOperation.SUMMARIZATION)
            
        client = self._get_client(provider)
        model = self._get_model_name(provider, ModelOperation.SUMMARIZATION)
        
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
                            context=context or "No additional context provided.",
                            results=results
                        )
                    }
                ],
                temperature=0.8,  # Slightly higher for creative forecasting                top_p=0.95,  # Focus on most likely tokens
                presence_penalty=0.2,  # Encourage diverse insights
                frequency_penalty=0.2,  # Avoid repetitive patterns
            )
            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"Error generating summary with {provider}: {e}")
            if provider != ModelProvider.OPENAI and ModelProvider.OPENAI in self.models:
                logging.warning(f"Falling back to OpenAI for summary generation")
                return self.generate_summary(query, results, context, provider=ModelProvider.OPENAI)
            raise
    
    def generate_chunks(
        self, 
        content: str,
        provider: Optional[ModelProvider] = None
    ) -> Dict[str, str]:
        """Generate chunks using the specified provider.
        
        Returns:
            Dict containing 'analysis' and 'context' keys:
            - analysis: The detailed chunk analysis
            - context: Generated contextual information for later use
        """
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
                temperature=0.1,  # Low temperature for consistent analysis
                presence_penalty=0.1,  # Slight penalty to avoid repetition
                frequency_penalty=0.1,  # Slight penalty for diverse analysis
                )
                
            # Parse the response to extract analysis and context
            result = response.choices[0].message.content
            
            # Split response into analysis and context sections
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

    @retry(wait=wait_random_exponential(min=1, max=40), stop=stop_after_attempt(3))
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
            logging.error(f"Error getting embeddings from {provider}: {e}")
            if provider != ModelProvider.OPENAI and ModelProvider.OPENAI in self.models:
                logging.warning("Falling back to OpenAI embeddings")
                return self.embedding_request(text, provider=ModelProvider.OPENAI)
            raise