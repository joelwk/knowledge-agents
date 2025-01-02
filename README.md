# Knowledge Agents

## Description
A text analysis framework implementing a three-stage pipeline architecture for processing and analyzing temporal data. The system combines multiple AI models (OpenAI, Grok, Venice.AI) to perform embedding generation, chunk analysis, and summary generation, with particular emphasis on temporal context preservation and semantic analysis.

### Key Capabilities
- **Temporal Intelligence**: 
  - Precise datetime handling across timezones
  - Time-aware context generation
  - Historical pattern analysis
  
- **Distributed Processing**:
  - Multi-provider model orchestration
  - Concurrent chunk processing
  - Batched operations with progress tracking

- **Adaptive Analysis**:
  - Dynamic provider selection
  - Automatic fallback mechanisms
  - Environment-aware execution (notebook/terminal)

- **Performance Monitoring (Optional)**:
  - Literal AI integration for comprehensive monitoring
  - Thread-level execution tracking
  - Provider performance metrics
  - Error pattern analysis

The framework is designed for robust handling of large-scale text analysis tasks, with built-in support for data validation, error recovery, and detailed operational logging. It provides a flexible foundation for building knowledge processing applications with temporal awareness.

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
- **Pipeline Architecture**:
  - Embedding Generation (OpenAI/Grok)
  - Chunk Analysis (OpenAI/Grok/Venice)
  - Summary Generation with temporal context
- **Provider Integration**:
  - Dynamic model selection and fallback
  - Standardized cross-provider responses
  - Concurrent batch processing

### Performance Monitoring
- **Literal AI Integration**:
  - Thread-level execution tracking
  - Step-by-step performance metrics
  - Provider usage patterns
  - Error rate monitoring
- **Monitoring Features**:
  - Automatic OpenAI instrumentation
  - Custom step tracking
  - Error pattern analysis
  - Resource utilization metrics

### Data Processing
- **Time-Aware Analysis**:
  - Historical pattern recognition
  - Temporal context preservation
- **Content Management**:
  - Semantic chunking with quality thresholds
  - Duplicate detection and filtering
  - Multi-format data handling (CSV/Parquet/Excel)

### Runtime Features
- **Adaptive Execution**:
  - Environment-aware (Notebook/Terminal)
  - Async processing with progress tracking
  - Configurable worker pools
- **Error Recovery**:
  - Exponential backoff retries
  - Provider fallback chains
  - Comprehensive logging system

### Analysis Capabilities
- **Signal Processing**:
  - Semantic search and retrieval
  - Pattern detection and analysis
  - Multi-source data integration
- **Contextual Analysis**:
  - Thread activity monitoring
  - Impact assessment metrics
  - Narrative evolution tracking

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

4. Set up monitoring (optional):
   - Get a Literal AI API key
   - Set the environment variable:
     ```python
     os.environ["LITERAL_API_KEY"] = "your-literal-api-key"
     ```
   - Or pass it directly to the run function:
     ```python
     chunks, summary = await run_knowledge_agents(
         query=query,
         process_new=True,
         providers=providers,
         monitor_api_key="your-literal-api-key"
     )
     ```

5. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

6. Navigate to and open `knowledge_workbench.ipynb`

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
   - Update `config_template.ini`
   - Add your API keys and model preferences

4. Set up monitoring (optional):
   ```bash
   export LITERAL_API_KEY="your-literal-api-key"
   ```

5. Run the main script:
   ```bash
   python model_ops.py
   ```

## Monitoring Integration
The framework includes comprehensive performance monitoring through Literal AI integration:

### Features
- Thread-level execution tracking
- Step-by-step performance metrics
- Provider usage patterns
- Error rate monitoring
- Resource utilization metrics

### Usage
1. Enable monitoring by setting the Literal AI API key:
   ```python
   os.environ["LITERAL_API_KEY"] = "your-literal-api-key"
   ```

2. Run with monitoring enabled:
   ```python
   chunks, summary = await run_knowledge_agents(
       query=query,
       process_new=True,
       providers=providers,
       monitor_api_key=os.getenv("LITERAL_API_KEY")
   )
   ```

3. Access monitoring data through the Literal AI dashboard:
   - View thread execution timelines
   - Analyze provider performance
   - Track error patterns
   - Monitor resource usage

### Monitored Operations
- Embedding generation
- Content retrieval
- Chunk analysis
- Summary generation
- Error handling and recovery

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
- Prompt Engineering Research: [Temporal-Aware Language Models for Temporal Knowledge Graph Question Answering](https://arxiv.org/pdf/2410.18959); used for designing temporal-aware prompts and multimodal forecasting capabilities