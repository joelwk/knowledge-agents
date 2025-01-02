from csv import writer
from model_ops import KnowledgeAgent, load_config, ModelProvider
import logging
from typing import List, Dict, Optional
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from tenacity import retry, wait_random_exponential, stop_after_attempt

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

CONFIG_PATH = './config.ini'
model_config, app_config = load_config(config_path=CONFIG_PATH)
agent = KnowledgeAgent(model_config)

class Article:
    """Data class for article information."""
    def __init__(self, thread_id: str, posted_date_time: str, text_clean: str, filename: str):
        self.thread_id = thread_id
        self.posted_date_time = posted_date_time
        self.text_clean = text_clean
        self.filename = filename

    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> 'Article':
        return cls(
            thread_id=data["thread_id"],
            posted_date_time=data["posted_date_time"],
            text_clean=data["text_clean"],
            filename=data["filename"]
        )

def load_data_from_csvs(directory: str) -> List[Article]:
    """Load articles from all CSVs in a given directory with improved error handling."""
    directory_path = Path(directory)
    if not directory_path.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    article_list: List[Article] = []
    csv_files = list(directory_path.glob("*_subset.csv"))
    
    if not csv_files:
        logger.warning(f"No matching CSV files found in {directory}")
        return article_list
    
    for file_path in tqdm(csv_files, desc="Loading CSV files"):
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
            required_columns = {"thread_id", "posted_date_time", "text_clean", "filename"}
            
            if not required_columns.issubset(df.columns):
                logger.error(f"Missing required columns in {file_path}")
                continue
                
            articles = [
                Article.from_dict(row) 
                for _, row in df.iterrows()
            ]
            article_list.extend(articles)
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
            continue
    
    return article_list

def get_relevant_content(
    library: str = '.',
    knowledge_base: str = '.',
    batch_size: int = 100,
    provider: Optional[ModelProvider] = None
) -> None:
    """Get and store embeddings for articles using the specified embedding provider.
    
    This function focuses solely on generating embeddings for similarity search.
    It should be used with embedding-specific models (OpenAI or Grok embedding models).
    
    Args:
        library: Path to the source data directory
        knowledge_base: Path to store the embedded knowledge base
        batch_size: Size of batches for processing
        provider: Embedding model provider (OpenAI or Grok)
    """
    kb_path = Path(knowledge_base)
    
    # Check if knowledge base already exists and has content
    if kb_path.exists() and kb_path.stat().st_size > 0:
        logger.info(f"Knowledge base '{knowledge_base}' already exists. Skipping.")
        return
    
    logger.info("Creating knowledge base with embeddings...")
    
    try:
        loaded_articles = load_data_from_csvs(library)
        
        if not loaded_articles:
            logger.warning("No articles found in the library.")
            # Create empty knowledge base file
            kb_path.touch()
            return
        
        # Process articles in batches
        with kb_path.open("w", newline='') as f_object:
            writer_object = writer(f_object)
            
            # Write header
            writer_object.writerow([
                "thread_id", 
                "posted_date_time", 
                "text_clean", 
                "embedding"
            ])
            
            # Process in batches with progress bar
            for i in tqdm(range(0, len(loaded_articles), batch_size), desc="Processing articles"):
                batch = loaded_articles[i:i + batch_size]
                results = process_article_batch(
                    batch,
                    batch_size=batch_size,
                    provider=provider
                )
                
                # Write batch results
                for result in results:
                    writer_object.writerow(result)
                    
        logger.info(f"Successfully created knowledge base at {knowledge_base}")
        
    except Exception as e:
        logger.error(f"Error creating knowledge base: {str(e)}")
        raise

@retry(wait=wait_random_exponential(min=1, max=40), stop=stop_after_attempt(3))
def process_article_batch(
    articles: List[Article],
    batch_size: int = 100,
    provider: Optional[ModelProvider] = None
) -> List[List]:
    """Process a batch of articles to get embeddings using the specified provider.
    
    This function should only be used with embedding-specific models.
    """
    results = []
    
    for i in range(0, len(articles), batch_size):
        batch = articles[i:i + batch_size]
        try:
            # Get embeddings for all texts in batch at once
            texts = [article.text_clean for article in batch]
            response = agent.embedding_request(texts, provider)
            
            # Create results
            for article, embedding in zip(batch, response.embedding):
                results.append([
                    article.thread_id,
                    article.posted_date_time,
                    article.text_clean,
                    embedding,
                ])
                
        except Exception as e:
            logger.error(f"Error processing batch {i//batch_size}: {str(e)}")
            raise
    
    return results