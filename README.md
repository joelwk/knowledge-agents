# knowledge-agents

## Description
A repository focused on implementing knowledge-based agents for data processing and analysis. This project provides tools and utilities for building intelligent agents that can process, analyze, and derive insights from various data sources.

## Supported Models
The project supports multiple AI model providers:
- **OpenAI**: Default provider for both completions and embeddings
  - Requires: `OPENAI_API_KEY`, `OPENAI_MODEL`, `OPENAI_EMBEDDING_MODEL`
- **Grok (X.AI)**: Alternative provider with its own embedding model
  - Optional: `GROK_API_KEY`, `GROK_MODEL`, `GROK_EMBEDDING_MODEL`
- **Venice.AI**: Additional model provider for completions
  - Optional: `VENICE_API_KEY`, `VENICE_MODEL`

Configure your preferred provider in `config.ini`. The system features automatic fallback to OpenAI if the primary provider fails, ensuring robust operation.

## Technical Features
### Model Operations
- **Multi-Provider Support**: Seamless integration with OpenAI, Grok, and Venice.AI
- **Automatic Fallback**: Graceful degradation to OpenAI if primary provider fails
- **Concurrent Processing**: Efficient parallel processing of text chunks
- **Context-Aware Chunking**: Intelligent text segmentation preserving semantic boundaries

### Inference Operations
- **Embedding Management**: Support for multiple embedding providers with automatic caching
- **Smart Retrieval**: Semantic search with duplicate detection and automatic expansion
- **Batch Processing**: Efficient handling of large text volumes through batched operations
- **Progress Tracking**: Built-in progress monitoring for long-running operations

### Error Handling
- **Retry Mechanism**: Exponential backoff for API calls
- **Graceful Degradation**: Automatic fallback to alternative providers
- **Comprehensive Logging**: Detailed error tracking and operation monitoring

## Running in Jupyter Notebook (knowledge_workbench)
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/knowledge-agents.git
   cd knowledge-agents
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure your model providers:
   - Copy `config_template.ini` to `config.ini`
   - Add your API keys and model preferences

4. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

5. Navigate to and open `knowledge_workbench.ipynb`

## Running from Terminal
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/knowledge-agents.git
   cd knowledge-agents
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure your model providers:
   - Copy `config_template.ini` to `config.ini`
   - Add your API keys and model preferences

4. Run the main script:
   ```bash
   python model_ops.py
   ```

## Data Gathering
For data collection functionality, you can utilize the data gathering tools from the [chanscope-lambda repository](https://github.com/joelwk/chanscope-lambda). If you prefer not to set up a Lambda function, you can use the `gather.py` script directly from that repository for data collection purposes.

### Using gather.py
1. Clone the chanscope-lambda repository
2. Navigate to the gather.py script
3. Follow the script's documentation for standalone data gathering functionality

## Prompt System
The `prompt.yaml` file is a crucial component that defines the system's interaction patterns and analytical capabilities. It contains two main sections:

### System Prompts
1. **Objective Analysis**
   - Handles complex forecasting tasks combining numerical and textual data
   - Performs structured analysis including numerical validation, contextual integration, and pattern recognition
   - Generates multimodal forecasts with confidence metrics and contextual validation

2. **Generate Chunks**
   - Specializes in processing and analyzing text segments
   - Performs temporal analysis, information extraction, and context generation
   - Maintains structured output format for consistency

### User Prompts
1. **Summary Generation**
   - Templates for comprehensive summaries with forecasting capabilities
   - Integrates numerical data with contextual information
   - Includes historical analysis, forecast generation, and risk assessment

2. **Text Chunk Summary**
   - Templates for analyzing discrete text segments
   - Extracts time series data and key information
   - Generates domain context, background knowledge, and assumptions

Each prompt type is designed to maintain temporal awareness, preserve numerical precision, and provide comprehensive contextual analysis. The system uses these prompts to ensure consistent, high-quality output across different analytical tasks.

## References
- Data Gathering Lambda: [chanscope-lambda](https://github.com/joelwk/chanscope-lambda)
- Prompt Engineering Research: [Temporal-Aware Language Models for Temporal Knowledge Graph Question Answering](https://arxiv.org/pdf/2410.18959) - Used for designing temporal-aware prompts and multimodal forecasting capabilities