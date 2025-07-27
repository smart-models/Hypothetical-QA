# Test Suite for Q&A Generation API

This directory contains a comprehensive test suite for the Q&A generation API, ensuring reliability, functionality, and quality of the question-answer generation system.

## 📁 File Structure

```
test/
├── README.md           # This documentation
├── pytest.ini         # Pytest configuration
├── test_qa_api.py      # Main test suite
└── test_chunks.json    # Test dataset (auto-generated if missing)
```

## 🚀 Quick Start

### Prerequisites
```bash
# Ensure Ollama is running
ollama serve

# Install required model
ollama pull qwen2.5:7b-instruct

# Install Python dependencies
pip install pytest fastapi pytest-cov requests
```

### Running Tests
```bash
# Navigate to project directory
cd "D:\Hypothetical Chunks Questions Answer"

# Run all tests
python -m pytest test/test_qa_api.py -v

# Run specific test
python -m pytest test/test_qa_api.py::test_qa_generation_basic_functionality -v

# Run with coverage report
python -m pytest test/test_qa_api.py --cov=hypothetical_qa_api --cov-report=html
```

## 🧪 Test Categories

### Core Functionality Tests
- **`test_qa_generation_basic_functionality`**: Tests basic Q&A generation workflow
- **`test_health_check_endpoint`**: Validates API health and status endpoints

### Input Validation Tests  
- **`test_parameter_validation`**: Tests parameter validation and error handling
- **`test_invalid_json_file_handling`**: Tests graceful handling of malformed input files

### Feature Tests
- **`test_custom_prompt_functionality`**: Tests custom prompt support
- **`test_qa_format_validation`**: Validates Q&A content format and structure

### Quality Assurance Tests
- **`test_quality_scoring_functionality`**: Tests quality assessment metrics
- **`test_processing_metadata_accuracy`**: Validates metadata accuracy and completeness

## 📊 Quality Benchmarks

The tests validate the following quality standards:

- **Response Structure**: Proper JSON structure with `chunks` and `metadata`
- **Q&A Format**: Generated content follows question-answer patterns
- **Quality Scoring**: Quality scores between 0.0 and 1.0
- **Metadata Completeness**: All required metadata fields present
- **Processing Time**: Reasonable response times for test datasets

## 🔧 Test Configuration

### Key Test Parameters
- **LLM Model**: `qwen2.5:7b-instruct`
- **Temperature**: `0.1` (for consistent results)
- **Max Retries**: `1` (for faster test execution)
- **Q&A Pairs**: Varies by test (1-2 pairs typically)

### Required Metadata Fields
The tests validate presence of these metadata fields:
- `total_chunks`, `number_qa`, `total_qa_pairs`
- `failed_parses`, `average_quality_score`, `quality_issues`
- `llm_model`, `temperature`, `context_window`
- `custom_prompt_used`, `source`, `processing_time`

## 🎯 Test Data

### Auto-Generated Test Data
If `test_chunks.json` doesn't exist, the test suite automatically creates a minimal dataset:

```json
{
  "chunks": [
    {
      "text": "This is a test chunk for Q&A generation.",
      "id": 1,
      "token_count": 10
    },
    {
      "text": "Another test chunk with more content to process.",
      "id": 2, 
      "token_count": 12
    }
  ]
}
```

## 🐛 Troubleshooting

### Common Issues

1. **Ollama Server Not Available**
   ```bash
   # Start Ollama server
   ollama serve
   ```

2. **Model Not Found**
   ```bash
   # Install required model
   ollama pull qwen2.5:7b-instruct
   ```

3. **Import Errors**
   ```bash
   # Ensure you're in the correct directory
   cd "D:\Hypothetical Chunks Questions Answer"
   
   # Check Python path
   python -c "import sys; print(sys.path)"
   ```

4. **Test Timeouts**
   - Check Ollama server is responsive
   - Ensure sufficient system resources
   - Verify model is properly loaded

### Debug Mode
```bash
# Run with detailed output
python -m pytest test/test_qa_api.py -v -s --tb=long

# Run specific failing test
python -m pytest test/test_qa_api.py::test_name -v -s
```

## 📈 Coverage and Quality

### Generate Coverage Report
```bash
# HTML report
python -m pytest test/test_qa_api.py --cov=hypothetical_qa_api --cov-report=html

# Terminal report
python -m pytest test/test_qa_api.py --cov=hypothetical_qa_api --cov-report=term-missing
```

### Test Markers
Tests can be marked for selective execution:
```bash
# Run only fast tests (if markers are added)
python -m pytest test/test_qa_api.py -m "not slow"
```

## 🔍 Test Design Principles

- **Isolation**: Each test is independent and can run standalone
- **Repeatability**: Tests use consistent parameters for reproducible results
- **Comprehensive Coverage**: Tests cover happy paths, edge cases, and error conditions
- **Clear Assertions**: Each test has specific, meaningful assertions
- **Documentation**: All tests include clear docstrings explaining their purpose

## 📝 Adding New Tests

When adding new tests, follow these conventions:

1. **Naming**: Use descriptive names like `test_<functionality>_<specific_case>`
2. **Documentation**: Include comprehensive docstrings
3. **Structure**: Follow AAA pattern (Arrange, Act, Assert)
4. **Cleanup**: Use appropriate fixtures and cleanup mechanisms
5. **Assertions**: Use specific assertions with clear error messages

Example:
```python
def test_new_functionality_edge_case(client):
    """
    Test new functionality with edge case scenario.
    
    Validates that the system properly handles [specific scenario]
    and returns expected results.
    """
    # Arrange
    test_data = prepare_test_data()
    
    # Act
    response = client.post("/endpoint", data=test_data)
    
    # Assert
    assert response.status_code == 200
    assert "expected_field" in response.json()
```

This test suite ensures the Q&A generation API maintains high quality and reliability across all supported use cases.
