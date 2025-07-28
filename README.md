![LLM Powered](https://img.shields.io/badge/LLM-Powered-green)
![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Latest-blue)
![Docker](https://img.shields.io/badge/Docker-Ready-blue)
![Ollama](https://img.shields.io/badge/Ollama-Compatible-orange)

# Hypothetical Chunks Questions Answer Generator

Hypothetical Chunks Questions Answer Generator is a specialized tool designed for Retrieval-Augmented Generation (RAG) systems that transforms text chunks into high-quality hypothetical question-answer pairs using Large Language Models (LLMs). The system intelligently analyzes input text and generates contextually relevant questions with comprehensive answers, enhancing RAG systems by creating synthetic data that improves retrieval performance and query understanding.

Built on modern FastAPI architecture with seamless Ollama integration, this production-ready solution leverages state-of-the-art language models to understand content semantics and generate relevant question-answer pairs. The system employs sophisticated prompt engineering and quality control mechanisms to ensure generated content maintains semantic accuracy and contextual relevance to the source material.

The system is designed for scalable deployment in RAG pipelines, vector database enrichment, and AI assistant. Whether you're processing knowledge base documents, technical documentation, or domain-specific content, the Hypothetical QA Generator provides a robust solution for creating synthetic question-answer pairs that improve retrieval performance at scale.

## Key Features

- **Hypothetical Q&A Generation**: Creates contextually relevant questions with comprehensive answers from text chunks for RAG systems
- **Content-Aware Processing**: Specialized prompts that adapt to different content types (scientific, technical, narrative, business, etc.)
- **LLM Integration**: Seamless integration with Ollama for local LLM deployment and model flexibility
- **Quality Control System**: Advanced parsing, validation, and quality scoring for generated content
- **Parallel Processing**: Concurrent chunk processing with configurable worker pools for optimal performance
- **Production-Ready API**: FastAPI-based REST interface with automatic documentation, validation, and error handling
- **Docker Support**: Complete containerization with both CPU and GPU deployment options
- **Comprehensive Testing**: Full test suite with pytest integration for reliability assurance
- **Flexible Configuration**: Extensive parameter control for model selection, generation settings, and processing options
- **Retry Mechanisms**: Robust error handling with configurable retry logic for network and generation failures
- **Detailed Metadata**: Comprehensive processing statistics, quality metrics, and performance analytics
- **Custom Prompt Support**: Ability to use custom prompt templates for specialized RAG use cases

## Table of Contents

- [How the Q&A Generation Algorithm Works](#how-the-qa-generation-algorithm-works)
  - [The Processing Pipeline](#the-processing-pipeline)
  - [Intelligent Content Analysis](#intelligent-content-analysis)
  - [Quality Control and Validation](#quality-control-and-validation)
  - [Comparison with Traditional Q&A Generation](#comparison-with-traditional-qa-generation)
- [Advantages of the Solution](#advantages-of-the-solution)
  - [Educational Value](#educational-value)
  - [Production Performance](#production-performance)
  - [Flexibility and Customization](#flexibility-and-customization)
- [Installation and Deployment](#installation-and-deployment)
  - [Prerequisites](#prerequisites)
  - [Getting the Code](#getting-the-code)
  - [Local Installation with Uvicorn](#local-installation-with-uvicorn)
  - [Docker Deployment (Recommended)](#docker-deployment-recommended)
  - [Ollama Setup](#ollama-setup)
- [Using the API](#using-the-api)
  - [API Endpoints](#api-endpoints)
  - [Example API Call](#example-api-call)
  - [Response Format](#response-format)
- [Configuration](#configuration)
- [Custom Prompt Templates](#custom-prompt-templates)
  - [Default Prompt Template](#default-prompt-template)
- [Testing](#testing)
  - [Running Tests](#running-tests)
  - [Test Coverage](#test-coverage)
- [Contributing](#contributing)

## How the Q&A Generation Algorithm Works

### The Processing Pipeline

The QA Generator implements a sophisticated multi-stage pipeline that combines advanced prompt engineering with LLM-powered content generation:

1. **Input Processing**: The API accepts JSON documents containing text chunks through the `/process-chunks/` endpoint
2. **Content Analysis**: Each text chunk is analyzed for content type, complexity, and educational value potential
3. **Intelligent Prompting**: The system applies specialized prompts based on content characteristics (scientific, narrative, technical, etc.)
4. **Parallel Generation**: Multiple chunks are processed concurrently using configurable worker pools for optimal throughput
5. **Q&A Extraction**: Generated content is parsed using sophisticated regex patterns to extract clean question-answer pairs
6. **Quality Validation**: Each Q&A pair undergoes comprehensive quality checks including completeness, relevance, and educational value
7. **Retry Logic**: Failed generations are automatically retried with exponential backoff for robust error handling
8. **Metadata Collection**: Detailed statistics are collected including processing times, quality scores, and generation success rates
9. **Response Assembly**: All Q&A pairs are combined with comprehensive metadata for client consumption

### Intelligent Content Analysis

The system employs advanced content understanding mechanisms:

```python
# Conceptual representation of the QA generation process
def qa_generation_process(chunks):
    processed_chunks = []
    
    # Parallel processing of chunks
    for chunk in chunks:
        # Analyze content type and apply appropriate prompt
        content_type = analyze_content_type(chunk.text)
        specialized_prompt = get_specialized_prompt(content_type)
        
        # Generate Q&A pairs with retry logic
        for attempt in range(max_retries):
            try:
                # Send to LLM with optimized prompt
                response = generate_with_llm(
                    prompt=specialized_prompt.format(
                        chunk=chunk.text,
                        num_pairs=n_qa_pairs
                    ),
                    model=llm_model,
                    temperature=temperature
                )
                
                # Parse and validate Q&A pairs
                qa_pairs = parse_qa_response(response)
                quality_score = calculate_quality_score(qa_pairs)
                
                if quality_score >= quality_threshold:
                    processed_chunks.extend(qa_pairs)
                    break
                    
            except (NetworkError, ParseError) as e:
                if attempt == max_retries - 1:
                    log_failed_chunk(chunk, e)
                else:
                    time.sleep(exponential_backoff(attempt))
    
    return {
        "chunks": processed_chunks,
        "metadata": compile_processing_metadata()
    }
```

This approach ensures high-quality educational content generation while maintaining robustness and performance in production environments.

### Quality Control and Validation

The QA Generator implements comprehensive quality assurance mechanisms:

- **Content Validation**: Ensures generated questions are complete, grammatically correct, and educationally valuable
- **Answer Verification**: Validates that answers are comprehensive, accurate, and properly formatted
- **Semantic Consistency**: Checks that Q&A pairs maintain semantic relationship with source content
- **Educational Standards**: Applies pedagogical principles to ensure questions test meaningful understanding
- **Format Compliance**: Validates proper question/answer formatting and structure

### Comparison with Traditional Q&A Generation

| Feature | Traditional Q&A Tools | Hypothetical Chunks QA Generator |
|---------|----------------------|-----------------------------------|
| Content Understanding | Rule-based or template-driven | LLM-powered semantic analysis |
| Question Quality | Basic extraction or simple patterns | Educationally optimized with quality scoring |
| Content Adaptation | One-size-fits-all approach | Specialized prompts for different content types |
| Error Handling | Limited retry mechanisms | Robust retry logic with exponential backoff |
| Scalability | Sequential processing | Parallel processing with worker pools |
| Customization | Fixed templates | Flexible prompt engineering and parameter control |
| Production Readiness | Basic functionality | Comprehensive monitoring, logging, and health checks |

## Advantages of the Solution

### Educational Value

The QA Generator creates pedagogically sound educational content:

- **Contextual Relevance**: Questions are generated based on actual content meaning, not just keyword matching
- **Educational Standards**: Prompts are designed to generate exam-worthy, educationally valuable questions
- **Content Type Awareness**: Different prompting strategies for narrative, scientific, technical, and other content types
- **Comprehension Focus**: Emphasizes understanding and analysis over simple recall
- **Quality Assurance**: Multi-layered validation ensures educational effectiveness

### Production Performance

The system is optimized for enterprise-grade deployment:

- **Concurrent Processing**: Parallel chunk processing with configurable worker pools
- **Robust Error Handling**: Comprehensive retry mechanisms and graceful failure management
- **Resource Optimization**: Intelligent allocation of CPU cores and memory usage
- **Health Monitoring**: Built-in health checks and comprehensive logging for production monitoring
- **Scalable Architecture**: Stateless API design enables horizontal scaling

### Flexibility and Customization

The QA Generator adapts to diverse educational use cases:

- **Model Selection**: Support for various LLM models through Ollama integration
- **Parameter Control**: Fine-tune generation temperature, context windows, and retry behavior
- **Custom Prompting**: Ability to provide custom prompt templates for specialized content
- **Integration Options**: REST API allows integration with any programming language or platform
- **Deployment Flexibility**: Run locally, in containers, or in cloud environments

## Installation and Deployment

### Prerequisites

- Docker and Docker Compose (for Docker deployment)
- Python 3.10+ (for local installation)
- Ollama installed and running (for LLM functionality)
- 4GB+ RAM recommended for optimal performance
- Optional: NVIDIA GPU for accelerated Ollama performance

### Getting the Code

Before proceeding with any installation method, clone or download the project:
```bash
# If using git
git clone <repository-url>
cd "Hypothetical Chunks Questions Answer"

# Or download and extract the project files
```

### Local Installation with Uvicorn

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Linux/Mac
   ```
   
   **For Windows users:**
   
   * Using Command Prompt:
   ```cmd
   venv\Scripts\activate.bat
   ```
   
   * Using PowerShell:
   ```powershell
   # If you encounter execution policy restrictions, run this once per session:
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
   
   # Then activate the virtual environment:
   venv\Scripts\Activate.ps1
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the FastAPI server:
   ```bash
   uvicorn hypothetical_qa_api:app --reload --host 0.0.0.0 --port 8000
   ```

4. The API will be available at `http://localhost:8000`.
   
   Access the API documentation and interactive testing interface at `http://localhost:8000/docs`.

### Docker Deployment (Recommended)

1. Create required directories for persistent storage:
   ```bash
   # Linux/macOS
   mkdir -p logs
   
   # Windows CMD
   mkdir logs
   
   # Windows PowerShell
   New-Item -ItemType Directory -Path logs -Force
   ```

2. Deploy with Docker Compose:

   **CPU-only deployment**:
   ```bash
   cd docker
   docker compose --profile cpu up -d
   ```

   **GPU-accelerated deployment** (requires NVIDIA GPU and Docker GPU support):
   ```bash
   cd docker
   docker compose --profile gpu up -d
   ```

   **Stopping the service**:
   ```bash
   # To stop CPU deployment
   docker compose --profile cpu down
   
   # To stop GPU deployment
   docker compose --profile gpu down
   ```

3. The API will be available at `http://localhost:8000`.

### Ollama Setup

#### For Docker Deployment

If you're using the Docker deployment method, Ollama is **automatically included** in the docker-compose configuration. The docker-compose.yml file defines dedicated `ollama-cpu` and `ollama-gpu` services that:

- Use the official `ollama/ollama:latest` image
- Are optimized for CPU or GPU operation respectively
- Have automatic model management and caching
- Are configured to work seamlessly with the QA API service

No additional Ollama setup is required when using Docker deployment.

#### For Local Installation

If you're using the local installation method with Uvicorn, you **must set up Ollama separately** before running the QA Generator:

1. **Install Ollama**:
   ```bash
   # Linux
   curl -fsSL https://ollama.ai/install.sh | sh
   
   # macOS
   brew install ollama
   
   # Windows
   # Download from https://ollama.ai/download
   ```

2. **Start Ollama service**:
   ```bash
   ollama serve
   ```

3. **Pull required model** (default: qwen2.5:7b-instruct):
   ```bash
   ollama pull qwen2.5:7b-instruct
   ```

The QA API will connect to Ollama at `http://localhost:11434` by default. You can change this by setting the `OLLAMA_BASE_URL` environment variable.

## Using the API

### API Endpoints

- **POST `/process-chunks/`**  
  Processes text chunks and generates question-answer pairs.
  
  **Parameters:**
  - `file`: JSON file containing text chunks to be processed
  - `llm_model`: LLM model to use for Q&A generation (string, default: "qwen2.5:7b-instruct")
  - `temperature`: Controls randomness in LLM output (float, default: 0.2)
  - `context_window`: Maximum context window size for LLM (integer, default: 24576)
  - `custom_prompt`: Optional custom prompt template for Q&A generation (string, optional)
  - `n_qa_pairs`: Number of Q&A pairs to generate per chunk (integer, default: 3, range: 1-10)
  - `max_retries`: Maximum retry attempts for failed generations (integer, default: 3, range: 1-10)
  
  **Expected JSON Input Format:**
  ```json
  {
    "chunks": [
      {
        "text": "First chunk of text content to generate questions from...",
        "id": 1,
        "token_count": 150
      },
      {
        "text": "Second chunk of text content...",
        "id": 2,
        "token_count": 200
      }
    ]
  }
  ```

- **GET `/`**  
  Health check endpoint returning service status, API version, and Ollama connectivity status.

### Example API Call

**Using cURL:**
```bash
# Basic usage
curl -X POST "http://localhost:8000/process-chunks/" \
  -F "file=@document.json" \
  -H "accept: application/json"

# With custom parameters
curl -X POST "http://localhost:8000/process-chunks/?llm_model=qwen2.5:7b-instruct&temperature=0.3&n_qa_pairs=5" \
  -F "file=@document.json" \
  -H "accept: application/json"
```

**Using Python:**
```python
import requests
import json

# API endpoint
api_url = 'http://localhost:8000/process-chunks/'
file_path = 'document.json'

# Prepare the document
document = {
    "chunks": [
        {
            "text": "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed.",
            "id": 1,
            "token_count": 25
        },
        {
            "text": "Deep learning uses neural networks with multiple layers to model and understand complex patterns in data.",
            "id": 2,
            "token_count": 20
        }
    ]
}

# Save to file
with open(file_path, 'w') as f:
    json.dump(document, f)

# Make the request
try:
    with open(file_path, 'rb') as f:
        files = {'file': (file_path, f, 'application/json')}
        params = {
            'llm_model': 'qwen2.5:7b-instruct',
            'temperature': 0.2,
            'n_qa_pairs': 3,
            'max_retries': 2
        }
        
        response = requests.post(api_url, files=files, params=params)
        response.raise_for_status()
        
        result = response.json()
        
        print(f"Generated {result['metadata']['total_qa_pairs']} Q&A pairs from {result['metadata']['total_chunks']} chunks")
        print(f"Average quality score: {result['metadata']['average_quality_score']:.2f}")
        print(f"Processing time: {result['metadata']['processing_time']:.2f}s")
        
        # Display sample Q&A pairs
        for i, qa in enumerate(result['chunks'][:3]):
            print(f"\nQ&A Pair {i+1}:")
            print(f"Source Chunk ID: {qa['id_source']}")
            print(f"Content: {qa['text'][:200]}...")
        
except Exception as e:
    print(f"Error: {e}")
```

### Response Format

A successful Q&A generation returns the following structure:

```json
{
  "chunks": [
    {
      "text": "Q: What is machine learning?\nA: Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed, allowing systems to automatically improve their performance on a specific task through data analysis.",
      "id": 1,
      "token_count": 45,
      "id_source": 1
    },
    {
      "text": "Q: How does deep learning differ from traditional machine learning?\nA: Deep learning uses neural networks with multiple layers to model and understand complex patterns in data, enabling automatic feature extraction and representation learning, whereas traditional machine learning often requires manual feature engineering.",
      "id": 2,
      "token_count": 52,
      "id_source": 2
    }
  ],
  "metadata": {
    "total_chunks": 2,
    "number_qa": 3,
    "total_qa_pairs": 6,
    "failed_parses": 0,
    "average_quality_score": 0.92,
    "quality_issues": {
      "empty_chunks": 0,
      "short_chunks": 0,
      "minimal_content": 0,
      "special_chars_only": 0,
      "no_output_generated": 0
    },
    "llm_model": "qwen2.5:7b-instruct",
    "temperature": 0.2,
    "context_window": 24576,
    "custom_prompt_used": false,
    "source": "document.json",
    "processing_time": 15.7
  }
}
```

## Configuration

The QA Generator can be configured through environment variables (for Docker deployments) or a local `.env` file. The table below lists configuration options with their default values:

| Variable | Description | Default |
|----------|-------------|---------|
| `OLLAMA_BASE_URL` | Base URL of the Ollama API server | `http://localhost:11434` |
| `LLM_MODEL` | Default LLM model used for Q&A generation | `qwen2.5:7b-instruct` |
| `TEMPERATURE` | Default sampling temperature for the LLM | `0.2` |
| `CONTEXT_WINDOW` | Maximum token window supplied to the LLM | `24576` |
| `DEFAULT_MAX_RETRIES` | Default maximum retry attempts for failed generations | `3` |
| `N_QA_PAIRS` | Default number of Q&A pairs to generate per chunk | `3` |
| `MIN_RETRIES` | Minimum allowed retry attempts | `1` |
| `MAX_RETRIES_LIMIT` | Maximum allowed retry attempts | `10` |
| `MAX_WORKERS` | Number of worker threads for parallel processing | `4` |
| `LOG_LEVEL` | Logging verbosity level | `INFO` |

## Custom Prompt Templates

The QA Generator allows you to customize the Q&A generation behavior by providing your own prompt template through the `custom_prompt` parameter. This gives you fine-grained control over how the LLM generates questions and answers.

### Default Prompt Template

Below is the default prompt template used by the QA Generator. You can use this as a starting point for creating your own custom prompts:

```python
PROMPT_TEMPLATE = '''
Act as an expert educator and assessment specialist. Your task is to create {num_pairs} high-quality, educationally valuable question-answer pairs from the provided text.

CONTENT ANALYSIS GUIDELINES:
Adapt your questions to the content type and domain. Consider the following specialized approaches:

- For NARRATIVE/LITERARY texts: Focus on plot development, character arcs, themes, symbolism, narrative techniques, conflicts, resolutions, and underlying messages
- For SCIENTIFIC/TECHNICAL texts: Emphasize methodology, hypotheses, experimental design, findings, limitations, applications, technical terminology, and implications for the field
- For ECONOMIC/BUSINESS texts: Target financial metrics, market trends, competitive strategies, ROI, stakeholder value, economic indicators, business models, industry disruptions, and strategic frameworks
- For POLITICAL/POLICY texts: Address legislation details, political positions, policy impacts, constituent effects, implementation challenges, partisan perspectives, and governance implications
- For HISTORICAL texts: Cover chronology, causation chains, historical context, key figures' roles, turning points, primary sources, historiographical debates, and long-term consequences

QUALITY CONTROLS - AVOID:
- Questions about minor procedural details or trivial facts
- Redundant questions covering the same concept
- Questions with obvious yes/no or single-word answers
- Grammar errors, missing articles, or incomplete sentences

QUALITY STANDARDS:
- Each question should test meaningful understanding of important concepts
- Questions should be exam-worthy and educationally valuable
- Prefer questions that require synthesis and analysis over recall
- Ensure variety in question types and difficulty levels

OUTPUT FORMAT:
- Provide your response as a clean sequence of questions and answers
- Each question must end with a question mark (?)
- Enclose each question in <Q> and </Q> tags
- Enclose each answer in <A> and </A> tags
- Do not add any extra text, explanations, or apologies outside the tags
- Use the same language as the source text

TEXT TO ANALYZE:
------------------------------------------------------------------------------------------
<text>
{chunk}
</text>
------------------------------------------------------------------------------------------
'''
```

### Important Note When Creating Custom Prompts

When creating your own custom prompt template, you **must** include the placeholders `{num_pairs}` and `{chunk}` as shown above. These placeholders will be replaced with the actual number of Q&A pairs requested and the text content to be processed.

Failure to include the proper placeholder format will cause the Q&A generation process to fail.

## Testing

The QA Generator includes a comprehensive test suite to ensure reliability and quality.

### Running Tests

```bash
# Run all tests
python -m pytest test/test_qa_api.py -v

# Run specific test
python -m pytest test/test_qa_api.py::test_qa_generation_basic_functionality -v

# Run with coverage report
python -m pytest test/test_qa_api.py --cov=hypothetical_qa_api --cov-report=html
```

### Test Coverage

The test suite includes:

- **Core Functionality Tests**: Basic Q&A generation workflow validation
- **Input Validation Tests**: Parameter validation and error handling
- **Feature Tests**: Custom prompt support and configuration options
- **Quality Assurance Tests**: Q&A format validation and quality scoring
- **Health Check Tests**: API health and service status validation
- **Metadata Tests**: Processing metadata accuracy and completeness

For detailed testing information, see the [test documentation](test/README.md).

## Contributing

Hypothetical Chunks Questions Answer Generator is an open-source project that welcomes contributions from the community. Your involvement helps improve educational technology for everyone.

We value contributions of all kinds:
- Bug fixes and performance improvements
- Documentation enhancements  
- New features and educational capabilities
- Test coverage improvements
- Integration examples and tutorials
- Educational prompt templates

If you're interested in contributing:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add or update tests as appropriate
5. Ensure all tests pass
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

Please ensure your code follows the existing style conventions and includes appropriate documentation.

For major changes, please open an issue first to discuss what you would like to change.

Happy Learning with QA Generator!

---
