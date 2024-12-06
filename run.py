import configparser
import json
import logging
import asyncio
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Union
from model_ops import load_config, KnowledgeAgent, EmbeddingProvider
from data_ops import prepare_data
from inference_ops import summarize_text
from embedding_ops import get_relevant_content
import nest_asyncio
import IPython

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

def load_run_config(config_path: str = './config_template.ini') -> Dict[str, str]:
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
        
        # Validate paths exist
        for key in ['ROOT_PATH', 'ALL_DATA', 'ALL_DATA_STRATIFIED_PATH']:
            path = Path(required_settings[key])
            if not path.exists():
                logger.warning(f"Path does not exist: {path}")
        
        return required_settings
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        raise

async def _run_knowledge_agents_async(
    query: str,
    process_new: bool = False,
    config_path: str = './config.ini',
    batch_size: int = 100,
    max_workers: Optional[int] = None,
    embedding_provider: Optional[str] = None
) -> Tuple[List[str], str]:
    """Async implementation of knowledge agents pipeline."""
    try:
        # Load configuration
        config = load_run_config(config_path)
        logger.info("Configuration loaded successfully")
        
        # Prepare data if requested
        if process_new:
            logger.info("Processing new data...")
            try:
                prepare_data()
            except Exception as e:
                logger.error(f"Error preparing data: {e}")
                raise
        
        # Get relevant content with batching
        try:
            get_relevant_content(
                library=config['ALL_DATA_STRATIFIED_PATH'],
                knowledge_base=config['KNOWLEDGE_BASE'],
                batch_size=batch_size,
                embedding_provider=embedding_provider
            )
        except Exception as e:
            logger.error(f"Error getting relevant content: {e}")
            raise
        
        # Generate summary
        try:
            chunks, response = await summarize_text(
                query=query,
                knowledge_base_path=config['KNOWLEDGE_BASE'],
                batch_size=batch_size,
                max_workers=max_workers
            )
            logger.info("Summary generated successfully")
            return chunks, response
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            raise
            
    except Exception as e:
        logger.error(f"Error in knowledge agent pipeline: {e}")
        raise

def run_knowledge_agents(
    query: str,
    process_new: bool = False,
    config_path: str = './config.ini',
    batch_size: int = 100,
    max_workers: Optional[int] = None,
    embedding_provider: Optional[Union[str, EmbeddingProvider]] = None
) -> Union[Tuple[List[str], str], asyncio.coroutine]:
    """Run knowledge agents pipeline in both notebook and script environments.
    
    This function detects the environment and handles the async execution appropriately.
    In a notebook, it will execute the coroutine immediately.
    In a script, it will return the coroutine for the event loop to execute.
    
    Args:
        query: The search query
        process_new: Whether to process new data
        config_path: Path to configuration file
        batch_size: Size of batches for processing
        max_workers: Maximum number of worker threads
        embedding_provider: Which embedding provider to use (openai or grok)
    
    Returns:
        In notebook: Tuple of (processed chunks, final summary)
        In script: Coroutine object
    """
    # Convert EmbeddingProvider enum to string if needed
    if isinstance(embedding_provider, EmbeddingProvider):
        embedding_provider = embedding_provider.value
        
    coroutine = _run_knowledge_agents_async(
        query=query,
        process_new=process_new,
        config_path=config_path,
        batch_size=batch_size,
        max_workers=max_workers,
        embedding_provider=embedding_provider
    )
    
    # If we're in a notebook, execute the coroutine immediately
    if IPython.get_ipython() is not None:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(coroutine)
    
    # Otherwise, return the coroutine for the script's event loop
    return coroutine

def main():
    """Main entry point with asyncio support."""
    import argparse
    parser = argparse.ArgumentParser(description='Run knowledge agents')
    parser.add_argument('query', type=str, help='Search query')
    parser.add_argument('--process-new', action='store_true', help='Process new data')
    parser.add_argument('--batch-size', type=int, default=100, help='Batch size for processing')
    parser.add_argument('--max-workers', type=int, help='Maximum number of worker threads')
    parser.add_argument('--embedding-provider', type=str, choices=['openai', 'grok'],
                      default=None, help='Embedding provider to use')
    
    args = parser.parse_args()
    
    try:
        # Get the coroutine and run it in the event loop
        coroutine = run_knowledge_agents(
            query=args.query,
            process_new=args.process_new,
            batch_size=args.batch_size,
            max_workers=args.max_workers,
            embedding_provider=args.embedding_provider
        )
        chunks, response = asyncio.run(coroutine)
        
        print("\nGenerated Summary:")
        print("-" * 80)
        print(response)
        print("-" * 80)
        
    except Exception as e:
        logger.error(f"Error running knowledge agents: {e}")
        raise

if __name__ == "__main__":
    main()