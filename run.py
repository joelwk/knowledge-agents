import configparser
import logging
import asyncio
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Union, Any
from model_ops import ModelProvider, ModelOperation, load_config
from data_ops import prepare_data
from inference_ops import summarize_text
from embedding_ops import get_relevant_content
from literalai import LiteralClient
import nest_asyncio
import IPython
from monitoring import get_monitored_ops
import time

# Enable nested asyncio for Jupyter notebooks
try:
    nest_asyncio.apply()
except Exception:
    pass

# Initialize logging with IPython-friendly format
class IPythonFormatter(logging.Formatter):
    """Custom formatter that detects if we're in a notebook."""
    def format(self, record):
        if IPython.get_ipython() is not None:
            # In notebook - use simple format
            self._style._fmt = "%(message)s"
        else:
            # In terminal - use detailed format
            self._style._fmt = "%(asctime)s - %(levelname)s - %(message)s"
        return super().format(record)

# Setup logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setFormatter(IPythonFormatter())
logger.addHandler(handler)
logger.setLevel(logging.INFO)

def load_run_config(config_path: str = './config.ini') -> Dict[str, str]:
    """Load and validate configuration settings."""
    try:
        config = configparser.ConfigParser()
        config.read(config_path)
        
        required_settings = {
            'ROOT_PATH': config['data']['ROOT_PATH'],
            'ALL_DATA': config['data']['ALL_DATA'],
            'ALL_DATA_STRATIFIED_PATH': config['data']['ALL_DATA_STRATIFIED_PATH'],
            'KNOWLEDGE_BASE': config['data']['KNOWLEDGE_BASE'],
            'SAMPLE_SIZE': config['configuration_params']['SAMPLE_SIZE'],
            'FILTER_DATE': config['configuration_params']['FILTER_DATE']
        }
        
        # Optional monitoring settings
        if 'monitoring' in config and 'LITERAL_API_KEY' in config['monitoring']:
            required_settings['LITERAL_API_KEY'] = config['monitoring']['LITERAL_API_KEY']
        
        # Validate paths exist
        for key in ['ROOT_PATH', 'ALL_DATA', 'ALL_DATA_STRATIFIED_PATH']:
            path = Path(required_settings[key])
            if not path.exists():
                logger.warning(f"Path does not exist: {path}")
        
        return required_settings
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        raise

def validate_providers(providers: Optional[Dict[ModelOperation, ModelProvider]] = None) -> Dict[ModelOperation, ModelProvider]:
    """Validate and set default providers for the three-model pipeline."""
    if providers is None:
        providers = {}
    
    # Ensure we have all required operations
    if ModelOperation.EMBEDDING not in providers:
        providers[ModelOperation.EMBEDDING] = ModelProvider.OPENAI
        logger.info(f"Using default embedding provider: {providers[ModelOperation.EMBEDDING]}")
    
    if ModelOperation.CHUNK_GENERATION not in providers:
        providers[ModelOperation.CHUNK_GENERATION] = ModelProvider.GROK
        logger.info(f"Using default chunking provider: {providers[ModelOperation.CHUNK_GENERATION]}")
    
    if ModelOperation.SUMMARIZATION not in providers:
        providers[ModelOperation.SUMMARIZATION] = ModelProvider.VENICE
        logger.info(f"Using default summarization provider: {providers[ModelOperation.SUMMARIZATION]}")
    
    # Validate embedding provider (only OpenAI and Grok support embeddings)
    if providers[ModelOperation.EMBEDDING] not in [ModelProvider.OPENAI, ModelProvider.GROK]:
        logger.warning(f"Unsupported embedding provider: {providers[ModelOperation.EMBEDDING]}. Falling back to OpenAI.")
        providers[ModelOperation.EMBEDDING] = ModelProvider.OPENAI
    
    return providers

def initialize_monitoring(
    api_key: Optional[str] = None,
) -> Optional[LiteralClient]:
    """Initialize Literal AI monitoring using the KnowledgeAgent's provider configuration."""
    if not api_key:
        logger.warning("No Literal AI API key provided. Monitoring will be disabled.")
        return None
    
    try:
        client = LiteralClient(api_key=api_key)
        # Only instrument OpenAI since that's what Literal AI supports
        client.instrument_openai()
        logger.info("OpenAI API monitoring instrumented")
        logger.info("Literal AI monitoring initialized successfully")
        return client
    except Exception as e:
        logger.error(f"Failed to initialize Literal AI monitoring: {e}")
        return None

async def _run_knowledge_agents_impl(
    query: str,
    process_new: bool = False,
    config_path: str = './config.ini',
    batch_size: int = 100,
    max_workers: Optional[int] = None,
    providers: Optional[Dict[ModelOperation, ModelProvider]] = None,
    monitor_api_key: Optional[str] = None
) -> Tuple[List[Dict[str, Any]], str]:
    """Implementation of the knowledge agents pipeline."""
    try:
        # Load configuration and initialize agent
        model_config, app_config = load_config(config_path)        
        # Initialize monitoring if API key is provided
        monitor = None
        thread_id = None
        if monitor_api_key or model_config.literal_api_key:
            monitor = get_monitored_ops(api_key=monitor_api_key or model_config.literal_api_key)
            if monitor and monitor.enabled:
                thread_id = monitor.create_thread(
                    query=query,
                    metadata={
                        "process_new": process_new,
                        "batch_size": batch_size,
                        "providers": {k.value: v.value for k, v in (providers or {}).items()}
                    }
                )
        
        # Validate providers
        providers = validate_providers(providers)
        
        # Track error rate for evaluation
        total_operations = 0
        failed_operations = 0
        
        # Wrap functions with monitoring if enabled
        get_relevant_content_func = get_relevant_content
        summarize_text_func = summarize_text
        prepare_data_func = prepare_data
        
        if monitor and monitor.enabled and thread_id:
            get_relevant_content_func = monitor.monitor_step(
                name="embedding_generation",
                step_type="llm",
                thread_id=thread_id
            )(get_relevant_content)
            
            summarize_text_func = monitor.monitor_step(
                name="summarization",
                step_type="llm",
                thread_id=thread_id
            )(summarize_text)
            
            prepare_data_func = monitor.monitor_step(
                name="data_preparation",
                step_type="run",
                thread_id=thread_id
            )(prepare_data)
        
        if process_new:
            logger.info("Processing new data...")
            start_time = time.time()
            total_operations += 1
            try:
                prepare_data_func(app_config)
            except Exception as e:
                failed_operations += 1
                raise
            finally:
                if monitor and monitor.enabled:
                    duration = time.time() - start_time
                    error_rate = failed_operations / total_operations if total_operations > 0 else 0
                    
                    # Log data preparation metrics
                    monitor.log_knowledge_agent_metrics(
                        query=query,
                        summary="",  # No summary at this stage
                        model_metrics={},  # No model metrics yet
                        analysis_metrics={
                            "duration": duration,
                            "error_rate": error_rate,
                            "total_operations": total_operations,
                            "failed_operations": failed_operations,
                            "stage": "data_preparation"
                        },
                        step_id=thread_id
                    )
        
        # Step 1: Generate embeddings using embedding model
        logger.info(f"Using {providers[ModelOperation.EMBEDDING]} for embeddings")
        start_time = time.time()
        total_operations += 1
        embedding_metrics = {}
        try:
            embedding_results = get_relevant_content_func(
                library=app_config.all_data_stratified_path,
                knowledge_base=app_config.knowledge_base,
                batch_size=batch_size,
                provider=providers[ModelOperation.EMBEDDING]
            )
            if isinstance(embedding_results, dict):
                embedding_metrics = embedding_results.get('metrics', {})
        except Exception as e:
            failed_operations += 1
            logger.error(f"Error getting relevant content: {e}")
            raise
        finally:
            if monitor and monitor.enabled:
                duration = time.time() - start_time
                error_rate = failed_operations / total_operations if total_operations > 0 else 0
                
                # Log embedding generation metrics
                monitor.log_knowledge_agent_metrics(
                    query=query,
                    summary="",  # No summary at this stage
                    model_metrics={
                        providers[ModelOperation.EMBEDDING].value: {
                            "duration": duration,
                            **embedding_metrics
                        }
                    },
                    analysis_metrics={
                        "error_rate": error_rate,
                        "total_operations": total_operations,
                        "failed_operations": failed_operations,
                        "stage": "embedding_generation"
                    },
                    step_id=thread_id
                )
        
        # Step 2 & 3: Generate chunks/context and final summary
        logger.info(f"Using {providers[ModelOperation.CHUNK_GENERATION]} for chunk analysis")
        logger.info(f"Using {providers[ModelOperation.SUMMARIZATION]} for final summary")
        start_time = time.time()
        total_operations += 1
        try:
            chunks, response = await summarize_text_func(
                query=query,
                knowledge_base_path=app_config.knowledge_base,
                batch_size=batch_size,
                max_workers=max_workers,
                providers=providers
            )
            
            # Calculate confidence score based on chunk quality
            chunk_confidence = sum(chunk.get('relevance_score', 0) for chunk in chunks) / len(chunks) if chunks else 0
            
            # Log and evaluate metrics
            if monitor and monitor.enabled:
                duration = time.time() - start_time
                error_rate = failed_operations / total_operations if total_operations > 0 else 0
                
                metrics = {
                    "duration": duration,
                    "error_rate": error_rate,
                    "confidence_score": chunk_confidence,
                    "chunks_generated": len(chunks),
                    "summary_length": len(response)
                }
                
                # Log metrics and evaluate against rules
                monitor.log_metrics(metrics, step_id=thread_id)
                monitor.evaluate_metrics(metrics)
                
                # Log final comprehensive metrics
                monitor.log_knowledge_agent_metrics(
                    query=query,
                    summary=response,
                    model_metrics={
                        providers[ModelOperation.EMBEDDING].value: {
                            "confidence": chunk_confidence,
                            **embedding_metrics
                        },
                        providers[ModelOperation.CHUNK_GENERATION].value: {
                            "chunks_generated": len(chunks)
                        },
                        providers[ModelOperation.SUMMARIZATION].value: {
                            "summary_length": len(response),
                            "duration": duration
                        }
                    },
                    analysis_metrics={
                        "error_rate": error_rate,
                        "total_operations": total_operations,
                        "failed_operations": failed_operations,
                        "stage": "complete",
                        "overall_duration": time.time() - start_time,
                        "confidence_score": chunk_confidence
                    },
                    step_id=thread_id
                )
            
            return chunks, response
        except Exception as e:
            failed_operations += 1
            logger.error(f"Error generating summary: {e}")
            raise
            
    except Exception as e:
        logger.error(f"Error in knowledge agents pipeline: {e}")
        raise

def run_knowledge_agents(
    query: str,
    process_new: bool = False,
    config_path: str = './config.ini',
    batch_size: int = 10,
    max_workers: Optional[int] = None,
    providers: Optional[Dict[ModelOperation, ModelProvider]] = None,
    monitor_api_key: Optional[str] = None
) -> Tuple[List[Dict[str, Any]], str]:
    """Run knowledge agents pipeline in both notebook and script environments.
    
    This function handles both notebook and script environments automatically.
    It will execute the async implementation appropriately in either context.
    
    Args:
        query: The search query
        process_new: Whether to process new data
        config_path: Path to configuration file
        batch_size: Size of batches for processing
        max_workers: Maximum number of worker threads
        providers: Dictionary mapping operations to providers
        monitor_api_key: Optional API key for monitoring
    
    Returns:
        Tuple of (processed chunks, final summary)
    """
    coroutine = _run_knowledge_agents_impl(
        query=query,
        process_new=process_new,
        config_path=config_path,
        batch_size=batch_size,
        max_workers=max_workers,
        providers=providers,
        monitor_api_key=monitor_api_key
    )
    
    # If we're in a notebook, use the current event loop
    if IPython.get_ipython() is not None:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(coroutine)
    
    # If we're in a script, use asyncio.run()
    return asyncio.run(coroutine)

def main():
    """Main entry point with support for three-model pipeline selection."""
    import argparse
    parser = argparse.ArgumentParser(description='Run knowledge agents with three-model pipeline')
    parser.add_argument('query', type=str, help='Search query')
    parser.add_argument('--process-new', action='store_true', help='Process new data')
    parser.add_argument('--batch-size', type=int, default=100, help='Batch size for processing')
    parser.add_argument('--max-workers', type=int, help='Maximum number of worker threads')
    parser.add_argument('--config', type=str, default='./config.ini', help='Path to config file')
    
    # Add provider arguments for each operation
    parser.add_argument('--embedding-provider', type=str, choices=['openai', 'grok'],
                      help='Provider for embeddings (only OpenAI and Grok supported)')
    parser.add_argument('--chunk-provider', type=str, choices=['openai', 'grok', 'venice'],
                      help='Provider for chunk generation and context analysis')
    parser.add_argument('--summary-provider', type=str, choices=['openai', 'grok', 'venice'],
                      help='Provider for final analysis and forecasting')
    
    args = parser.parse_args()
    
    try:
        # Build providers dictionary from arguments
        providers = {}
        if args.embedding_provider:
            providers[ModelOperation.EMBEDDING] = ModelProvider(args.embedding_provider)
        if args.chunk_provider:
            providers[ModelOperation.CHUNK_GENERATION] = ModelProvider(args.chunk_provider)
        if args.summary_provider:
            providers[ModelOperation.SUMMARIZATION] = ModelProvider(args.summary_provider)
        
        chunks, response = run_knowledge_agents(
            query=args.query,
            process_new=args.process_new,
            config_path=args.config,
            batch_size=args.batch_size,
            max_workers=args.max_workers,
            providers=providers
        )
        
        print("\nGenerated Summary:")
        print("-" * 80)
        print(response)
        print("-" * 80)
        
    except Exception as e:
        logger.error(f"Error running knowledge agents: {e}")
        raise

if __name__ == "__main__":
    main()