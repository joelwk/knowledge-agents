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

Configure your preferred provider in `config.ini`. The system will automatically fall back to OpenAI if the primary provider fails.

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

## References
- Data Gathering Lambda: [chanscope-lambda](https://github.com/joelwk/chanscope-lambda)