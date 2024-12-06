import ast
import pandas as pd
from scipy import spatial
import tiktoken
import concurrent.futures
from tqdm import tqdm
from typing import List, Dict, Any, Optional, Tuple
import configparser
from model_ops import KnowledgeAgent, load_config, EmbeddingProvider
from embedding_ops import get_relevant_content
import logging
import numpy as np

# Initialize logging
logger = logging.getLogger(__name__)

config_path = './config_template.ini'
model_config, app_config = load_config(config_path=config_path)
agent = KnowledgeAgent(model_config)

def strings_ranked_by_relatedness(
    query: str,
    df: pd.DataFrame,
    top_n: int = 100,
    embedding_provider: Optional[str] = None
) -> List[str]:
    """Returns a list of strings sorted from most related to least, based on cosine similarity."""
    try:
        query_embedding_response = agent.embedding_request(
            text=query,
            provider=embedding_provider
        )
        query_embedding = query_embedding_response.embedding
        
        # Convert string embeddings back to lists if they're stored as strings
        if isinstance(df["embedding"].iloc[0], str):
            df["embedding"] = df["embedding"].apply(ast.literal_eval)
        
        # Convert embeddings to numpy array for vectorized operations
        embeddings = np.vstack(df["embedding"].values)
        
        # Calculate cosine similarities in one vectorized operation
        similarities = 1 - spatial.distance.cdist([query_embedding], embeddings, metric='cosine')[0]
        
        # Get indices of top N similar items
        top_indices = np.argsort(similarities)[::-1][:top_n]
        
        return df.iloc[top_indices]["text"].tolist()
        
    except Exception as e:
        logger.error(f"Error ranking strings by relatedness: {e}")
        raise

def create_chunks(text: str, n: int, tokenizer=None) -> List[Dict[str, Any]]:
    """Creates chunks of text, preserving sentence boundaries where possible."""
    if tokenizer is None:
        tokenizer = tiktoken.get_encoding("cl100k_base")
    
    # First, split text into sentences
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        # Add period back to sentence
        sentence = sentence + '.'
        sentence_tokens = tokenizer.encode(sentence)
        sentence_length = len(sentence_tokens)
        
        # If single sentence is too long, split it
        if sentence_length > n:
            # If we have a current chunk, add it first
            if current_chunk:
                chunks.append({
                    "text": ' '.join(current_chunk),
                    "token_count": current_length
                })
                current_chunk = []
                current_length = 0
            
            # Split long sentence into chunks
            words = sentence.split()
            temp_chunk = []
            temp_length = 0
            
            for word in words:
                word_tokens = tokenizer.encode(word + ' ')
                if temp_length + len(word_tokens) > n:
                    if temp_chunk:
                        chunks.append({
                            "text": ' '.join(temp_chunk),
                            "token_count": temp_length
                        })
                    temp_chunk = [word]
                    temp_length = len(word_tokens)
                else:
                    temp_chunk.append(word)
                    temp_length += len(word_tokens)
            
            if temp_chunk:
                chunks.append({
                    "text": ' '.join(temp_chunk),
                    "token_count": temp_length
                })
                
        # If adding this sentence would exceed chunk size, start new chunk
        elif current_length + sentence_length > n:
            if current_chunk:
                chunks.append({
                    "text": ' '.join(current_chunk),
                    "token_count": current_length
                })
            current_chunk = [sentence]
            current_length = sentence_length
        else:
            current_chunk.append(sentence)
            current_length += sentence_length
    
    # Add final chunk if exists
    if current_chunk:
        chunks.append({
            "text": ' '.join(current_chunk),
            "token_count": current_length
        })
    
    return chunks

def retrieve_unique_strings(
    query: str,
    library_df: pd.DataFrame,
    required_count: int = 25,
    embedding_provider: Optional[str] = None
) -> List[str]:
    """Fetches a specified number of unique top strings, expanding retrieval if duplicates exist."""
    try:
        strings = strings_ranked_by_relatedness(
            query,
            library_df,
            top_n=required_count * 2,  # Get more initially to account for duplicates
            embedding_provider=embedding_provider
        )
        
        # Use dict to maintain order while removing duplicates
        unique_strings = list(dict.fromkeys(strings))
        
        if len(unique_strings) < required_count:
            logger.info(f"Only {len(unique_strings)} unique strings found, retrieving additional...")
            more_strings = strings_ranked_by_relatedness(
                query,
                library_df,
                top_n=required_count * 4,  # Try with even more
                embedding_provider=embedding_provider
            )
            # Extend unique strings while preserving order
            unique_strings = list(dict.fromkeys(unique_strings + more_strings))
        
        return unique_strings[:required_count]
        
    except Exception as e:
        logger.error(f"Error retrieving unique strings: {e}")
        raise

async def summarize_text(
    query: str,
    knowledge_base_path: str = ".",
    batch_size: int = 5,
    max_workers: Optional[int] = None,
    embedding_provider: Optional[str] = None
) -> Tuple[List[Dict[str, Any]], str]:
    """Summarizes related texts from the knowledge base."""
    summary_prompt = """Summarize this text of social media dialog. Extract any key points with reasoning.\n\nContent:"""
    
    try:
        # Load and prepare knowledge base
        library_df = pd.read_csv(knowledge_base_path)
        if len(library_df) == 0:
            logger.info("No papers found, downloading first.")
            get_relevant_content(query)
            library_df = pd.read_csv(knowledge_base_path)
        
        library_df.columns = ["thread_id", "posted_date_time", "text", "embedding"]
        
        # Get unique related strings
        unique_strings = retrieve_unique_strings(
            query,
            library_df,
            embedding_provider=embedding_provider
        )
        
        logger.info("Processing text chunks")
        tokenizer = tiktoken.get_encoding("cl100k_base")
        all_chunks = []
        
        # Process each text into chunks
        for text in unique_strings:
            chunks = create_chunks(text, 1000, tokenizer)  # Increased chunk size
            all_chunks.extend(chunks)
        
        if not all_chunks:
            logger.warning("No valid chunks found to process")
            return [], ""
        
        logger.info(f"Summarizing {len(all_chunks)} chunks of text")
        
        # Configure worker pool
        if max_workers is None:
            max_workers = min(8, len(all_chunks))
        
        summaries = []
        
        # Process chunks in batches
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            
            # Submit all chunks for processing
            for chunk in all_chunks:
                future = executor.submit(
                    agent.generate_chunks,
                    chunk["text"],
                    summary_prompt
                )
                futures.append(future)
            
            # Collect results with progress bar
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                try:
                    result = future.result()
                    if result and len(result.strip()) > 0:
                        summaries.append(result)
                except Exception as e:
                    logger.error(f"Error processing chunk: {e}")
                    continue
        
        if not summaries:
            logger.warning("No summaries generated")
            return all_chunks, ""
        
        # Combine summaries and generate final summary
        combined_summary = "\n\n".join(summaries)
        
        try:
            logger.info("Generating final summary")
            response = agent.generate_summary(query, combined_summary)
            return all_chunks, response
            
        except Exception as e:
            logger.error(f"Error generating final summary: {e}")
            # Return the most relevant summaries as fallback
            return all_chunks, "\n\n".join(summaries[:3])
            
    except Exception as e:
        logger.error(f"Error in summarization pipeline: {e}")
        raise