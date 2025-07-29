"""
Test suite for Q&A Generation API.

This module contains comprehensive tests for the Q&A generation API,
including functionality, validation, error handling, and quality assurance tests.
"""

import json
import os
import sys
import tempfile

import pytest
from fastapi.testclient import TestClient

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hypothetical_qa_api import app

# Directory for test data files
TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "test_data")


@pytest.fixture(scope="session", autouse=True)
def setup_test_data():
    """
    Ensure test data files exist for testing.

    Creates minimal test data file if it doesn't exist.
    This fixture runs automatically before all tests.
    """
    test_chunks_path = os.path.join(TEST_DATA_DIR, "test_chunks.json")

    if not os.path.exists(test_chunks_path):
        # Create minimal test data if file doesn't exist
        test_data = {
            "chunks": [
                {
                    "text": "This is a test chunk for Q&A generation.",
                    "id": 1,
                    "token_count": 10,
                },
                {
                    "text": "Another test chunk with more content to process.",
                    "id": 2,
                    "token_count": 12,
                },
            ]
        }
        with open(test_chunks_path, "w", encoding="utf-8") as f:
            json.dump(test_data, f, indent=2)

    yield  # Run the tests


@pytest.fixture
def client():
    """Create a test client for the FastAPI application."""
    with TestClient(app) as test_client:
        yield test_client


def test_qa_generation_basic_functionality(client):
    """
    Test basic Q&A generation functionality with test_chunks.json.

    Validates the core API functionality including:
    - API response structure
    - Generated Q&A content
    - Metadata completeness
    """
    test_file = os.path.join(TEST_DATA_DIR, "test_chunks.json")

    with open(test_file, "rb") as f:
        response = client.post(
            "/process-chunks/",
            files={"file": ("test_chunks.json", f, "application/json")},
            params={
                "llm_model": "qwen2.5:7b-instruct",
                "temperature": 0.1,
                "n_qa_pairs": 2,
                "max_retries": 1,
            },
        )

    assert response.status_code == 200, f"API returned error: {response.text}"

    data = response.json()

    # Validate basic response structure
    assert "chunks" in data, "Response missing 'chunks' key"
    assert "metadata" in data, "Response missing 'metadata' key"

    chunks = data["chunks"]
    metadata = data["metadata"]

    # Validate Q&A chunks
    assert isinstance(chunks, list), "'chunks' should be a list"
    assert len(chunks) > 0, "No Q&A pairs were generated"

    # Validate metadata structure for Q&A API
    required_metadata_fields = [
        "total_chunks",
        "number_qa",
        "total_qa_pairs",
        "failed_parses",
        "average_quality_score",
        "quality_issues",
        "llm_model",
        "temperature",
        "context_window",
        "custom_prompt_used",
        "source",
        "processing_time",
    ]

    for field in required_metadata_fields:
        assert field in metadata, f"Missing required metadata field: {field}"

    # Validate Q&A chunk structure
    for chunk in chunks:
        assert "text" in chunk, "Q&A chunk missing 'text' field"
        assert "id" in chunk, "Q&A chunk missing 'id' field"
        assert "token_count" in chunk, "Q&A chunk missing 'token_count' field"
        assert "id_source" in chunk, "Q&A chunk missing 'id_source' field"

        assert len(chunk["text"]) > 0, "Q&A text should not be empty"
        assert chunk["id"] > 0, "Q&A id should be positive"
        assert chunk["id_source"] > 0, "Source id should be positive"

    print(
        f"Successfully generated {len(chunks)} Q&A pairs from "
        f"{metadata['total_chunks']} chunks"
    )
    print(f"Quality score: {metadata['average_quality_score']:.2f}")
    print(f"Processing time: {metadata['processing_time']}s")


def test_health_check_endpoint(client):
    """
    Test the health check endpoint functionality.

    Ensures the API is running and all required health information
    is available.
    """
    response = client.get("/")
    assert response.status_code == 200

    data = response.json()
    assert "status" in data
    assert "version" in data
    assert "ollama_status" in data
    assert "ollama_url" in data


def test_invalid_json_file_handling(client):
    """
    Test graceful handling of invalid JSON files.

    Ensures the API properly handles malformed input files
    without crashing.
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write('{"invalid": "json"}')
        temp_path = f.name

    try:
        with open(temp_path, "rb") as f:
            response = client.post(
                "/process-chunks/",
                files={"file": ("invalid.json", f, "application/json")},
            )

        # Should handle gracefully, not necessarily fail
        assert response.status_code in [200, 422]
    finally:
        os.unlink(temp_path)


def test_parameter_validation(client):
    """
    Test parameter validation for Q&A API endpoints.

    Ensures invalid parameters are properly rejected with
    appropriate error codes.
    """
    test_file = os.path.join(TEST_DATA_DIR, "test_chunks.json")

    with open(test_file, "rb") as f:
        # Test with invalid n_qa_pairs (too high)
        response = client.post(
            "/process-chunks/",
            files={"file": ("test_chunks.json", f, "application/json")},
            params={"n_qa_pairs": 15},  # Should be > 10 to trigger validation
        )

    assert response.status_code == 422  # Validation error


def test_custom_prompt_functionality(client):
    """
    Test custom prompt functionality.

    Validates that custom prompts are properly accepted and used
    in the Q&A generation process.
    """
    test_file = os.path.join(TEST_DATA_DIR, "test_chunks.json")

    custom_prompt = (
        "Generate exactly 2 educational questions and answers from this text: {chunk}"
    )

    with open(test_file, "rb") as f:
        response = client.post(
            "/process-chunks/",
            files={"file": ("test_chunks.json", f, "application/json")},
            params={"custom_prompt": custom_prompt, "n_qa_pairs": 2},
        )

    assert response.status_code == 200
    data = response.json()

    # Check that custom prompt was used
    assert data["metadata"]["custom_prompt_used"] is True


def test_qa_format_validation(client):
    """
    Test that generated Q&A pairs have proper format.

    Validates that the generated content follows recognizable
    question-and-answer patterns.
    """
    test_file = os.path.join(TEST_DATA_DIR, "test_chunks.json")

    with open(test_file, "rb") as f:
        response = client.post(
            "/process-chunks/",
            files={"file": ("test_chunks.json", f, "application/json")},
            params={"n_qa_pairs": 1},
        )

    assert response.status_code == 200
    data = response.json()

    # Validate Q&A format - should contain questions and answers
    for qa_item in data["chunks"]:
        text = qa_item["text"]
        text_lower = text.lower()

        # Should contain indicators of Q&A format
        assert "?" in text, "Q&A should contain questions"

        # Check for various answer indicators (more flexible)
        answer_indicators = [
            "answer:",
            "a:",
            "ans:",
            "answer is",
            "the answer",
            "response:",
            "reply:",
            "solution:",
            "explanation:",
        ]
        question_indicators = [
            "question:",
            "q:",
            "que:",
            "query:",
            "what",
            "how",
            "why",
            "when",
            "where",
        ]

        # Must have either answer indicators OR question indicators
        # to be considered Q&A format
        has_answer_indicators = any(
            keyword in text_lower for keyword in answer_indicators
        )
        has_question_indicators = any(
            keyword in text_lower for keyword in question_indicators
        )

        assert has_answer_indicators or has_question_indicators, (
            f"Q&A should contain recognizable Q&A format indicators. "
            f"Text: {text[:100]}..."
        )


def test_quality_scoring_functionality(client):
    """
    Test that quality scoring is working correctly.

    Validates the quality assessment metrics and ensures
    they fall within expected ranges.
    """
    test_file = os.path.join(TEST_DATA_DIR, "test_chunks.json")

    with open(test_file, "rb") as f:
        response = client.post(
            "/process-chunks/",
            files={"file": ("test_chunks.json", f, "application/json")},
            params={"n_qa_pairs": 2},
        )

    assert response.status_code == 200
    data = response.json()

    # Quality score should be between 0 and 1
    quality_score = data["metadata"]["average_quality_score"]
    assert 0.0 <= quality_score <= 1.0, f"Quality score {quality_score} out of range"

    # Quality issues should be tracked
    quality_issues = data["metadata"]["quality_issues"]
    assert isinstance(quality_issues, dict), "Quality issues should be a dict"
    assert "empty_chunks" in quality_issues
    assert "short_chunks" in quality_issues
    assert "minimal_content" in quality_issues


def test_processing_metadata_accuracy(client):
    """
    Test that processing metadata accurately reflects the operation.

    Ensures that all metadata fields contain accurate information
    about the processing operation.
    """
    test_file = os.path.join(TEST_DATA_DIR, "test_chunks.json")

    # Load original file to get expected chunk count
    with open(test_file, "r", encoding="utf-8") as f:
        original_data = json.load(f)
    expected_chunks = len(original_data["chunks"])

    with open(test_file, "rb") as f:
        response = client.post(
            "/process-chunks/",
            files={"file": ("test_chunks.json", f, "application/json")},
            params={"n_qa_pairs": 2},
        )

    assert response.status_code == 200
    data = response.json()

    # Check metadata accuracy
    metadata = data["metadata"]
    assert metadata["total_chunks"] == expected_chunks
    assert metadata["number_qa"] == 2  # Should match requested n_qa_pairs
    assert metadata["source"] == "test_chunks.json"
    assert isinstance(metadata["processing_time"], (int, float))
    assert metadata["processing_time"] > 0
