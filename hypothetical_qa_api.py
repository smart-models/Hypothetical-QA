import json
import time
import logging
import requests
import tiktoken
import random
import re
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
from pydantic import BaseModel
from pydantic_settings import BaseSettings
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse
from tqdm import tqdm
from contextlib import asynccontextmanager
from logging.handlers import RotatingFileHandler
from functools import lru_cache


# Configurazione Settings
class Settings(BaseSettings):
    # LLM Settings
    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "qwen2.5:7b-instruct"
    temperature: float = 0.2
    context_window: int = 24576  # 24k context window

    # Default Settings
    default_max_retries: int = 3
    n_qa_pairs: int = 3
    min_retries: int = 1
    max_retries_limit: int = 10

    # Concurrency Settings
    # Optimal number for CPU usage without blocking API
    max_workers: int = 4

    class Config:
        case_sensitive = False
        extra = "ignore"

    @property
    def ollama_api_generate_url(self) -> str:
        """URL for the Ollama generate API endpoint"""
        base = self.ollama_base_url.rstrip("/")
        return f"{base}/api/generate"

    @property
    def ollama_api_tags_url(self) -> str:
        """URL for the Ollama tags API endpoint"""
        base = self.ollama_base_url.rstrip("/")
        return f"{base}/api/tags"

    @property
    def ollama_api_pull_url(self) -> str:
        """URL for the Ollama pull API endpoint"""
        base = self.ollama_base_url.rstrip("/")
        return f"{base}/api/pull"


@lru_cache()
def get_settings():
    """Get application settings.

    This function returns a cached instance of the Settings class.

    Returns:
        Settings: The application settings object.

    """
    return Settings()


# Prompt template per generare QA
QA_PROMPT_TEMPLATE = """
Act as an expert technical content analyst with deep knowledge across multiple domains.
Analyze the following text (delimited by lines containing only dashes and tags) carefully and generate EXACTLY {num_pairs} question-answer pairs (no more, no less) that demonstrate deep comprehension of the content.
Make sure to count your output and verify you've created the exact number requested.

QUESTION QUALITY HIERARCHY:
- PRIORITIZE: Conceptual understanding, principles, methodologies, and key insights
- EMPHASIZE: "Why" and "how" questions over simple "what" questions
- FOCUS: Questions that test comprehension and application, not memorization
- AVOID: Trivial facts, overly granular details, or single-word answers

QUESTION DISTRIBUTION (aim for balanced mix):
- 40% Conceptual (principles, frameworks, theoretical understanding)
- 30% Analytical (relationships, implications, cause-and-effect)  
- 20% Applied (examples, case studies, practical applications)
- 10% Factual (key definitions, important data points, essential facts)

ANSWER REQUIREMENTS:
- Provide sufficient context for standalone understanding
- Include specific examples when available in the text
- Aim for 2-4 sentences unless complexity requires more
- Ensure answers are complete, accurate, and self-contained
- Use proper grammar with articles (a, an, the) where needed

CONTEXT-AWARE INSTRUCTIONS:
Adapt your questions to the text type:
- For NARRATIVE/LITERARY texts: Focus on plot development, character arcs, themes, symbolism, narrative techniques, conflicts, resolutions, and underlying messages
- For SCIENTIFIC/TECHNICAL texts: Emphasize methodology, hypotheses, experimental design, findings, limitations, applications, technical terminology, and implications for the field
- For ECONOMIC/BUSINESS texts: Target financial metrics, market trends, competitive strategies, ROI, stakeholder value, economic indicators, business models, industry disruptions, and strategic frameworks
- For POLITICAL/POLICY texts: Address legislation details, political positions, policy impacts, constituent effects, implementation challenges, partisan perspectives, and governance implications
- For HISTORICAL texts: Cover chronology, causation chains, historical context, key figures' roles, turning points, primary sources, historiographical debates, and long-term consequences
- For MEDICAL/HEALTH texts: Focus on symptoms, diagnoses, treatment protocols, patient outcomes, clinical trials, side effects, epidemiological data, and healthcare implications
- For LEGAL/REGULATORY texts: Examine precedents, statutory interpretation, compliance requirements, case law, legal reasoning, jurisdictional issues, and practical applications
- For EDUCATIONAL/ACADEMIC texts: Target learning objectives, pedagogical approaches, theoretical frameworks, research methodologies, academic debates, and knowledge applications
- For TECHNOLOGICAL/IT texts: Emphasize system architecture, algorithms, performance metrics, security considerations, scalability, integration challenges, and future developments
- For ENVIRONMENTAL/CLIMATE texts: Address ecological impacts, sustainability metrics, climate data, conservation strategies, stakeholder interests, policy recommendations, and scientific consensus
- For PHILOSOPHICAL/ETHICAL texts: Explore arguments, logical structures, ethical frameworks, thought experiments, counterarguments, practical implications, and philosophical traditions
- For JOURNALISTIC/NEWS texts: Focus on the 5W1H, sources cited, potential biases, broader context, stakeholder reactions, fact vs. opinion, and societal impact
- For BIOGRAPHICAL texts: Cover life events, achievements, challenges overcome, historical context, personal relationships, legacy, and character development
- For INSTRUCTIONAL/HOW-TO texts: Target step sequences, required materials, skill prerequisites, common pitfalls, tips for success, safety considerations, and outcome expectations
- For MARKETING/ADVERTISING texts: Analyze target audience, value propositions, persuasion techniques, brand messaging, call-to-action elements, and competitive positioning

QUALITY CONTROLS - AVOID:
- Questions about minor procedural details or trivial facts
- Redundant questions covering the same concept
- Questions with obvious yes/no or single-word answers
- Grammar errors, missing articles, or incomplete sentences
- Questions that could only be answered by someone with the exact text in front of them

QUALITY STANDARDS:
- Each question should test meaningful understanding of important concepts
- Questions should be exam-worthy and educationally valuable
- Prefer questions that require synthesis and analysis over recall
- Ensure variety in question types and difficulty levels
- Review for clarity, completeness, and educational value

OUTPUT FORMAT:
- Provide your response as a clean sequence of questions and answers
- Each question must end with a question mark (?)
- Enclose each question in <Q> and </Q> tags
- Enclose each answer in <A> and </A> tags
- Do not add any extra text, explanations, or apologies outside the tags
- Use the same language as the source text
- Review your output before submitting to ensure grammar and completeness

ENHANCED EXAMPLES:

Conceptual Questions:
<Q>What is the fundamental principle behind the Build-Measure-Learn feedback loop in lean startup methodology?</Q>
<A>The fundamental principle is to minimize the time and resources spent building products that customers don't want by rapidly testing hypotheses through small experiments, measuring customer response, and learning from the results to inform the next iteration.</A>

Analytical Questions:
<Q>How does the concept of validated learning differ from traditional business planning approaches?</Q>
<A>Validated learning relies on empirical data from real customer interactions and behavior to test business assumptions, while traditional planning often depends on market research, forecasts, and theoretical analysis that may not reflect actual customer needs or market conditions.</A>

Applied Questions:
<Q>What specific techniques did IMVU use to test their instant messaging product assumptions with customers?</Q>
<A>IMVU brought potential customers into their office to try the product directly, conducted numerous conversations with users, and observed that teenagers and tech early adopters were more likely to engage while mainstream users found it too unfamiliar and resisted downloading or sharing it.</A>

Factual Questions:
<Q>What are the three main engines of growth identified in the lean startup model?</Q>
<A>The three main engines of growth are the viral engine (customers bring in new customers through word-of-mouth or product sharing), the sticky engine (focuses on customer retention and reducing churn), and the paid engine (uses advertising or sales to acquire customers profitably).</A>

FINAL CHECK:
Before submitting, verify:
1. You have created EXACTLY {num_pairs} question-answer pairs
2. Questions focus on important concepts rather than trivial details
3. Answers are complete and grammatically correct
4. You have a good mix of question types
5. Each question tests distinct knowledge or understanding

TEXT TO ANALYZE:
------------------------------------------------------------------------------------------
<text>
{chunk}
</text>
------------------------------------------------------------------------------------------
"""


# Pydantic Models for API
class Chunk(BaseModel):
    text: str
    id: int
    token_count: int


class QAItem(BaseModel):
    text: str
    id: int
    token_count: int
    id_source: int


class Metadata(BaseModel):
    total_chunks: int
    number_qa: int
    total_qa_pairs: int
    failed_parses: int
    average_quality_score: float
    quality_issues: Dict[str, int] = {
        "empty_chunks": 0,
        "short_chunks": 0,
        "minimal_content": 0,
        "special_chars_only": 0,
        "no_output_generated": 0,
    }
    llm_model: str
    temperature: float
    context_window: int
    custom_prompt_used: bool
    source: str
    processing_time: float


class ResponseModel(BaseModel):
    chunks: List[QAItem]
    metadata: Metadata


# Create logs directory if it doesn't exist.
logs_dir = Path("logs")
logs_dir.mkdir(exist_ok=True)

# Configure logging.
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Get the logger.
logger = logging.getLogger(__name__)

# Create a file handler for error logs.
error_log_path = logs_dir / "errors.log"
file_handler = RotatingFileHandler(
    error_log_path,
    maxBytes=10485760,  # 10 MB.
    backupCount=5,  # Keep 5 backup logs.
    encoding="utf-8",
)

# Set the file handler to only log errors and critical messages.
file_handler.setLevel(logging.ERROR)

# Create a formatter.
formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(message)s - %(pathname)s:%(lineno)d"
)
file_handler.setFormatter(formatter)

# Add the handler to the logger.
logger.addHandler(file_handler)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager that initializes models before startup
    and cleans up resources on shutdown.
    """
    logger.info("QA API starting up...")

    # First check if the Ollama server is reachable at all
    # Check server with verbose logging (this is only done once at startup)
    server_reachable = check_ollama_server_reachable(verbose=False)

    if server_reachable:
        # Only try to ensure the model is available if the server is reachable
        logger.info(
            f"Ollama server at {get_settings().ollama_base_url} is reachable. Checking for required QA models..."
        )
        ensure_ollama_model(get_settings().llm_model, get_settings().llm_model)
    else:
        # Use critical level for more visibility in logs
        logger.critical("⚠️ WARNING: OLLAMA SERVER NOT AVAILABLE ⚠️")
        logger.critical(
            "The QA API is starting with LIMITED FUNCTIONALITY. "
            "QA generation features will NOT WORK "
            "until the Ollama server becomes available."
        )
        logger.warning(
            "Please ensure Ollama is running and accessible at: "
            + get_settings().ollama_base_url
        )

    # Store server status for post-startup message
    app.state.ollama_available = server_reachable

    yield

    # Cleanup on shutdown
    logger.info("QA API shutting down...")


app = FastAPI(
    title="Q&A Generation API",
    description="An API to generate question-answer pairs from text chunks using Ollama.",
    version="0.5.0",
    lifespan=lifespan,
)


# Helper functions
def get_token_count(text: str) -> int:
    """Estimate the number of tokens in a given text.

    It first tries to use the 'cl100k_base' encoding from tiktoken.
    If that fails, it falls back to a simple word count.

    Args:
        text (str): The text to count tokens for.

    Returns:
        int: The estimated number of tokens.

    """
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception:
        # Fallback for other models
        return len(text.split())


def calculate_honest_quality_score(
    chunks: List[dict], results: List[dict], quality_issues: Dict[str, int]
) -> float:
    """Calculate an honest quality score that reflects real issues in chunks and results.

    Args:
        chunks: List of original chunks
        results: List of processing results
        quality_issues: Dictionary with count of detected problems

    Returns:
        float: Quality score between 0.0 and 1.0 that reflects actual quality
    """
    if not chunks or not results:
        return 0.0

    total_chunks = len(chunks)

    # Base score from traditional successful parsing
    successful_parses = sum(
        1 for r in results if r.get("metrics", {}).get("parsing_successful", False)
    )
    base_score = successful_parses / total_chunks if total_chunks > 0 else 0

    # Calculate penalties based on detected problems
    total_issues = sum(quality_issues.values())
    issue_penalty = min(total_issues / total_chunks, 0.8)  # Max 80% penalty

    # Additional penalty for no_output_generated (more severe)
    no_output_penalty = (
        quality_issues.get("no_output_generated", 0) / total_chunks
    ) * 0.3

    # Final score
    final_score = max(0.0, base_score - issue_penalty - no_output_penalty)

    return round(final_score, 2)


def check_ollama_server_reachable(
    ollama_base_url: str = None, timeout: int = 5, verbose: bool = False
):
    """
    Check if the Ollama server is reachable before attempting any model operations.

    Args:
        ollama_base_url: Optional override of the base Ollama URL. If None, uses the value from settings.
        timeout: Timeout in seconds for the connection attempt
        verbose: Whether to log detailed messages about server connectivity

    Returns:
        bool: True if the server is reachable, False otherwise
    """
    settings = get_settings()
    base_url = (ollama_base_url or settings.ollama_base_url).rstrip("/")

    # Try to connect to the root endpoint - Ollama shows "Ollama is running" on root
    try:
        # First try the root endpoint
        response = requests.get(f"{base_url}", timeout=timeout)
        if response.status_code == 200:
            if verbose:
                logger.info(f"Ollama server at {base_url} is reachable")
            return True

        # If that fails, try the /api/tags endpoint which should exist
        response = requests.get(f"{base_url}/api/tags", timeout=timeout)
        if response.status_code == 200:
            if verbose:
                logger.info(f"Ollama server at {base_url} is reachable via /api/tags")
            return True

        # If both fail but server responds, log the error
        logger.error(
            f"Ollama server at {base_url} returned unexpected status code {response.status_code}"
        )
        return False
    except requests.exceptions.ConnectionError:
        logger.error(
            f"Could not connect to Ollama server at {base_url}. Is Ollama running?"
        )
        return False
    except requests.exceptions.Timeout:
        logger.error(f"Timeout connecting to Ollama server at {base_url}")
        return False
    except Exception as e:
        logger.error(f"Error checking Ollama server availability: {e}")
        return False


def check_ollama_model(model_name: str, ollama_base_url: str = None):
    """
    Checks if a specific Ollama model is available locally via the API.

    Args:
        model_name: The name of the model to check (e.g., "qwen2.5:7b-instruct").
        ollama_base_url: The base URL of the Ollama API.

    Returns:
        True if the model is available locally, False otherwise.
    """
    settings = get_settings()

    # Use provided base URL or get from settings
    base_url = (ollama_base_url or settings.ollama_base_url).rstrip("/")
    api_url = f"{base_url}/api/tags"
    try:
        response = requests.get(api_url, timeout=10)
        response.raise_for_status()

        data = response.json()
        models = data.get("models", [])

        if not isinstance(models, list):
            logger.error(
                f"Unexpected format from Ollama API /api/tags. Expected a list under 'models'. Response: {data}"
            )
            return False

        # Check if any model name exactly matches our model_name
        for model in models:
            if isinstance(model, dict) and model.get("name") == model_name:
                return True

        return False

    except requests.exceptions.ConnectionError:
        logger.error(
            f"Could not connect to Ollama API at {ollama_base_url or settings.ollama_base_url}. Is Ollama running?"
        )
        return False
    except requests.exceptions.Timeout:
        logger.error(
            f"Timeout connecting to Ollama API at {ollama_base_url or settings.ollama_base_url}"
        )
        return False
    except requests.exceptions.RequestException as e:
        logger.error(f"Error checking for Ollama model: {e}")
        return False


def pull_ollama_model(
    model_name: str,
    ollama_base_url: str = None,
    stream: bool = False,
):
    """
    Triggers Ollama to pull a model using the API.

    Args:
        model_name: The name of the model to pull (e.g., "qwen2.5:7b-instruct").
        ollama_base_url: The base URL of the Ollama API.
        stream: Whether to process the response as a stream (True) or wait for completion (False).

    Returns:
        True if the pull request was successful, False otherwise.
    """
    settings = get_settings()

    # Use provided base URL or get from settings
    base_url = (ollama_base_url or settings.ollama_base_url).rstrip("/")
    api_url = f"{base_url}/api/pull"

    # Use the stream parameter as specified by the documentation
    # If stream=False in the API request, Ollama will wait until download completes and return single response
    payload = {"model": model_name, "stream": stream}
    logger.info(f"Pulling QA model '{model_name}' from Ollama...")

    try:
        # Always use stream=True for requests to allow processing response in chunks
        response = requests.post(api_url, json=payload, stream=True)
        response.raise_for_status()

        # If API stream=True, we'll get multiple status updates
        if stream:
            for line in response.iter_lines():
                if not line:
                    continue

                try:
                    status = json.loads(line.decode("utf-8"))
                    logger.info(f"Pull status: {status.get('status', 'unknown')}")

                    # Show progress percentage for downloads
                    if (
                        status.get("status", "").startswith("downloading")
                        and "total" in status
                        and "completed" in status
                        and status["total"] > 0
                    ):
                        progress = (status["completed"] / status["total"]) * 100
                        logger.info(f"Download progress: {progress:.1f}%")

                    if status.get("status") == "success":
                        return True
                except json.JSONDecodeError:
                    continue

            # If we got here without returning True, the stream ended without success
            logger.error("Model pull stream ended without success status")
            return False

        # If API stream=False, we'll get a single response at the end
        else:
            # Even with API stream=False, we still need to process the response
            last_status = None
            for line in response.iter_lines():
                if line:
                    try:
                        status = json.loads(line.decode("utf-8"))
                        last_status = status
                    except json.JSONDecodeError:
                        continue

            # Check final status
            if last_status and last_status.get("status") == "success":
                logger.info(f"QA model '{model_name}' pulled successfully")
                return True
            else:
                logger.error(f"Failed to pull QA model. Final status: {last_status}")
                return False

    except requests.exceptions.ConnectionError:
        logger.error(
            f"Could not connect to Ollama API at {ollama_base_url or settings.ollama_base_url}. Is Ollama running?"
        )
        return False
    except requests.exceptions.RequestException as e:
        logger.error(f"Error pulling Ollama model: {e}")
        if hasattr(e, "response") and e.response is not None:
            logger.error(f"Response status code: {e.response.status_code}")
            logger.error(f"Response text: {e.response.text}")
        return False


def ensure_ollama_model(model_name: str, fallback_model: str = None) -> str:
    """
    Ensures an Ollama model is available locally, attempting to pull it if not.
    First checks if the Ollama server is reachable before attempting any operations.
    If pulling fails and a fallback model is provided, it will verify the fallback is available.

    Args:
        model_name: The name of the model to ensure is available
        fallback_model: Optional fallback model to use if the requested model can't be pulled

    Returns:
        The name of the model that is available to use (either the requested or fallback model)
    """
    # First check if the Ollama server is reachable at all
    if not check_ollama_server_reachable(verbose=False):
        logger.warning(
            f"Ollama server is not reachable. Cannot ensure QA model '{model_name}' is available."
        )
        # Return the requested model name even though we can't verify it
        # This allows the application to start even if Ollama is not available
        return model_name

    # Check if model exists
    logger.info(f"Checking if QA model '{model_name}' is available locally...")
    if not check_ollama_model(model_name):
        logger.warning(
            f"QA model '{model_name}' not found locally. Attempting to pull..."
        )

        # Try to pull the model
        if pull_ollama_model(model_name, stream=False):
            logger.info(f"Successfully pulled QA model '{model_name}'")
            return model_name

        # If pull fails and we have a fallback model
        if fallback_model and fallback_model != model_name:
            logger.warning(
                f"Failed to pull QA model '{model_name}'. Trying fallback model '{fallback_model}'"
            )

            # Check if fallback model exists
            if not check_ollama_model(fallback_model):
                logger.warning(
                    f"Fallback QA model '{fallback_model}' not found locally. Attempting to pull..."
                )

                # Try to pull the fallback model
                if pull_ollama_model(fallback_model, stream=False):
                    logger.info(
                        f"Successfully pulled fallback QA model '{fallback_model}'"
                    )
                    return fallback_model
                else:
                    logger.error(
                        f"Failed to pull fallback QA model '{fallback_model}'. QA processing may fail."
                    )
                    # Return the fallback model name anyway, as that's our best option
                    return fallback_model
            else:
                logger.info(
                    f"Fallback QA model '{fallback_model}' is available locally"
                )
                return fallback_model
        else:
            # No fallback provided or fallback is the same as requested model
            logger.error(
                f"Failed to pull QA model '{model_name}' and no valid fallback available. QA processing may fail."
            )
            return model_name
    else:
        logger.info(f"QA model '{model_name}' is available locally")
        return model_name


def parse_and_validate_qa(
    response_text: str, expected_pairs: int, chunk_id: int
) -> Tuple[str, Dict[str, Any]]:
    """Parse and validate Q&A pairs from raw LLM text.

    It uses a primary regex method and a fallback line-splitting method to extract
    question-answer pairs. It also calculates metrics about the parsing process.

    Args:
        response_text (str): The raw text response from the LLM.
        expected_pairs (int): The number of Q&A pairs expected.
        chunk_id (int): The ID of the chunk being processed.

    Returns:
        Tuple[str, Dict[str, Any]]: A tuple containing the formatted Q&A string
                                     and a dictionary of metrics.

    """
    metrics = {
        "parsing_successful": False,
        "pairs_found": 0,
        "quality_score": 0.0,
        "parsing_method": "none",
        "issues": [],
    }
    qa_pairs = []

    # 1. Primary Method: Regex to find Q&A pairs enclosed in <Q> and <A> tags.
    regex = re.compile(r"<Q>(.*?)</Q>\s*<A>(.*?)</A>", re.IGNORECASE | re.DOTALL)
    matches = regex.findall(response_text)

    if matches:
        metrics["parsing_method"] = "regex_tags"
        for question, answer in matches:
            question = question.strip()
            answer = answer.strip()
            if question and answer:
                qa_pairs.append(f"{question}\n{answer}")

    # 2. Fallback Method: Simple line-based splitting for less structured output
    if not qa_pairs:
        metrics["parsing_method"] = "fallback_split"
        lines = [line.strip() for line in response_text.split("\n") if line.strip()]
        for i in range(0, len(lines) - 1, 2):
            question = lines[i]
            answer = lines[i + 1]
            if question.endswith("?") and not answer.endswith("?"):
                qa_pairs.append(f"{question}\n{answer}")

    if qa_pairs:
        metrics["parsing_successful"] = True
        metrics["pairs_found"] = len(qa_pairs)
        quality_ratio = min(1.0, len(qa_pairs) / expected_pairs)
        metrics["quality_score"] = round(quality_ratio, 2)
        if len(qa_pairs) < expected_pairs:
            metrics["issues"].append(f"found_{len(qa_pairs)}_expected_{expected_pairs}")

        return "\n".join(qa_pairs), metrics
    else:
        metrics["issues"].append("no_qa_pairs_extracted")
        logger.debug(
            f"Could not extract any Q&A pairs for chunk {chunk_id} using any method."
        )
        return "", metrics


def generate_corrective_qa(
    chunk: str,
    model: str,
    temperature: float,
    context_window: int,
    chunk_id: int,
    correction_type: str,
    existing_question: str = None,
    max_retries: int = 3,
) -> Tuple[str, str]:
    """Generate a corrective Q&A pair using the main prompt with specific feedback.

    This function is used to fix issues like missing questions, missing answers,
    or corrupted pairs from a previous generation attempt.

    Args:
        chunk (str): The source text chunk.
        model (str): The name of the LLM model to use.
        temperature (float): The generation temperature.
        context_window (int): The context window size for the LLM.
        chunk_id (int): The ID of the chunk being processed.
        correction_type (str): The type of correction needed ('missing_question',
                               'missing_answer', or 'corrupted_pair').
        existing_question (str, optional): The existing question if the answer is
                                           missing. Defaults to None.
        max_retries (int, optional): The maximum number of retry attempts.
                                     Defaults to 3.

    Returns:
        Tuple[str, str]: A tuple containing the corrected question and answer.

    """
    settings = get_settings()
    api_url = settings.ollama_api_generate_url

    # Feedback specifici per ogni caso
    feedback_messages = {
        "missing_question": (
            "FOCUS SPECIFICALLY ON CREATING A HIGH-QUALITY QUESTION with its corresponding answer. "
            "The previous generation attempt failed to produce a valid question."
        ),
        "missing_answer": (
            f"FOCUS SPECIFICALLY ON CREATING A COMPREHENSIVE ANSWER for this question: '{existing_question}'. "
            f"The previous generation attempt produced the question but failed to generate a valid answer."
        ),
        "corrupted_pair": (
            "The previous generation attempt produced corrupted or incomplete Q&A. "
            "Focus on creating ONE complete, high-quality question-answer pair."
        ),
    }

    # Usa sempre il prompt principale con feedback specifico
    corrective_prompt = QA_PROMPT_TEMPLATE.format(num_pairs=1, chunk=chunk).replace(
        "IMPORTANT: Analyze the following text carefully and generate EXACTLY {num_pairs} question-answer pairs (no more, no less)",
        f"IMPORTANT: Analyze the following text carefully and generate EXACTLY 1 question-answer pair. "
        f"{feedback_messages[correction_type]}",
    )

    headers = {"Content-Type": "application/json"}
    data = {
        "model": model,
        "prompt": corrective_prompt,
        "temperature": temperature,
        "options": {"num_ctx": context_window},
        "stream": False,
    }

    # Retry logic con backoff esponenziale
    for retry_count in range(max_retries):
        try:
            response = requests.post(
                api_url, headers=headers, data=json.dumps(data), timeout=60
            )
            response.raise_for_status()
            response_json = response.json()
            full_response = response_json.get("response", "").strip()

            if full_response:
                # Parse la risposta per estrarre Q e A
                qa_result, metrics = parse_and_validate_qa(
                    full_response, expected_pairs=1, chunk_id=chunk_id
                )

                if metrics["parsing_successful"] and metrics["pairs_found"] == 1:
                    # Estrai domanda e risposta
                    pairs = qa_result.strip().split("\n")
                    if len(pairs) >= 2:
                        # Using word boundary (\b) to prevent cutting off first letters of words like 'Alice'
                        question = re.sub(
                            r"^(Q|Question)\b[:\s]*", "", pairs[0], flags=re.IGNORECASE
                        ).strip()
                        answer = re.sub(
                            r"^(A|Answer)\b[:\s]*", "", pairs[1], flags=re.IGNORECASE
                        ).strip()
                        return question, answer

            if retry_count == max_retries - 1:
                logger.debug(
                    f"All {max_retries} correction attempts failed for chunk {chunk_id}"
                )
                return "", ""

            # Exponential backoff con jitter
            delay = (2**retry_count) + random.uniform(0, 1)
            time.sleep(delay)

        except requests.exceptions.RequestException as e:
            logger.debug(
                f"Correction attempt {retry_count + 1}/{max_retries} failed for chunk {chunk_id}: {e}"
            )
            if retry_count == max_retries - 1:
                return "", ""
    return "", ""


def generate_qa_from_chunk(
    chunk: str,
    model: str,
    prompt: str,
    temperature: float,
    context_window: int,
    chunk_id: int,
    num_pairs: int = 3,
    max_retries: int = 3,
) -> Tuple[str, Dict[str, Any]]:
    """Generate Q&A pairs from a text chunk using Ollama.

    This function sends a request to the Ollama API to generate Q&A pairs
    from the given text chunk. It includes retry logic for network errors and
    for cases where the LLM does not return the expected number of pairs.

    Args:
        chunk (str): The text chunk to generate Q&A from.
        model (str): The name of the LLM model to use.
        prompt (str): The prompt template to use for generation.
        temperature (float): The generation temperature.
        context_window (int): The context window size for the LLM.
        chunk_id (int): The ID of the chunk being processed.
        num_pairs (int, optional): The number of Q&A pairs to generate.
                                   Defaults to 3.
        max_retries (int, optional): The maximum number of retry attempts.
                                     Defaults to 3.

    Returns:
        Tuple[str, Dict[str, Any]]: A tuple containing the formatted Q&A string
                                     and a dictionary of metrics.

    """
    settings = get_settings()
    api_url = settings.ollama_api_generate_url
    timeout = 360 # seconds

    # Format the prompt with both num_pairs and the chunk text
    formatted_prompt = prompt.format(num_pairs=num_pairs, chunk=chunk)

    headers = {"Content-Type": "application/json"}
    data = {
        "model": model,
        "prompt": formatted_prompt,
        "temperature": temperature,
        "options": {"num_ctx": context_window},
        "stream": False,  # We will handle the full response
    }

    # Use settings for network retries
    network_retries = max_retries or settings.default_max_retries
    base_delay = 2  # Seconds.

    # Retry logic with explicit check for number of QA pairs
    qa_attempts = 0
    max_qa_attempts = max_retries

    while qa_attempts < max_qa_attempts:
        # Retry configuration for network errors
        last_exception = None

        for attempt in range(network_retries):
            try:
                response = requests.post(
                    api_url, headers=headers, data=json.dumps(data), timeout=timeout
                )

                if response.status_code == 200:
                    response_json = response.json()
                    full_response = response_json.get("response", "").strip()

                    if not full_response:
                        # If we got an empty response, log and retry.
                        logger.warning(
                            f"Empty response received from OLLAMA for chunk {chunk_id} (attempt {attempt + 1}/{network_retries})"
                        )
                        if attempt == network_retries - 1:
                            raise Exception(
                                "LLM returned empty response after multiple retries"
                            )
                    else:
                        qa_result, metrics = parse_and_validate_qa(
                            full_response, expected_pairs=num_pairs, chunk_id=chunk_id
                        )

                        # Check if we got the expected number of QA pairs
                        if (
                            metrics["pairs_found"] == num_pairs
                            or qa_attempts == max_qa_attempts - 1
                        ):
                            return qa_result, metrics

                        # Adjust prompt to emphasize exact count more strongly
                        enhanced_prompt = prompt.format(
                            num_pairs=num_pairs, chunk=chunk
                        ).replace(
                            "EXACTLY {num_pairs}",
                            f"EXACTLY {num_pairs} (CURRENTLY YOU PRODUCED {metrics['pairs_found']}, NEED {num_pairs})",
                        )
                        data["prompt"] = enhanced_prompt
                        qa_attempts += 1
                        break  # Break out of the network retry loop and try a new QA generation attempt

                else:
                    # Log the error and prepare for retry.
                    logger.warning(
                        f"Error response from OLLAMA for chunk {chunk_id} (attempt {attempt + 1}/{network_retries}): "
                        f"Status {response.status_code}, Response: {response.text[:200]}..."
                    )
                    if attempt == network_retries - 1:
                        # On last attempt, raise the exception.
                        raise Exception(
                            f"Error generating QA pairs: {response.status_code}"
                        )

            except requests.exceptions.Timeout as e:
                logger.warning(
                    f"Timeout error connecting to OLLAMA for chunk {chunk_id} (attempt {attempt + 1}/{network_retries}): {str(e)}"
                )
                last_exception = e

            except requests.exceptions.ConnectionError as e:
                logger.warning(
                    f"Connection error to OLLAMA for chunk {chunk_id} (attempt {attempt + 1}/{network_retries}): {str(e)}"
                )
                last_exception = e

            except requests.exceptions.RequestException as e:
                logger.warning(
                    f"Request error to OLLAMA for chunk {chunk_id} (attempt {attempt + 1}/{network_retries}): {str(e)}"
                )
                last_exception = e

            except Exception as e:
                logger.warning(
                    f"Unexpected error during OLLAMA request for chunk {chunk_id} (attempt {attempt + 1}/{network_retries}): {str(e)}"
                )
                last_exception = e

            # Only sleep if we're going to retry.
            if attempt < network_retries - 1:
                # Exponential backoff with jitter.
                delay = base_delay * (2**attempt) + random.uniform(0, 1)
                logger.info(f"Retrying chunk {chunk_id} in {delay:.2f} seconds...")
                time.sleep(delay)

        # If we've exhausted all network retries, log the error and raise an exception.
        if last_exception:
            logger.error(
                f"Failed to connect to OLLAMA for chunk {chunk_id} after {network_retries} attempts"
            )
            raise Exception(
                f"Failed to connect to OLLAMA after {network_retries} attempts: {str(last_exception)}"
            )

    # This part should not be reached in normal operation
    return "Error: Failed to get response from LLM after multiple retries.", {
        "error": "max_retries_exceeded"
    }


@app.get("/")
async def health_check():
    """Check the health status of the API service.

    Returns:
        dict: A dictionary containing:
            - status: Current health status of the service (healthy or degraded)
            - version: Current API version
            - ollama_status: Status of the Ollama server connectivity
            - ollama_url: URL of the Ollama server
    """
    # Check current Ollama connectivity
    ollama_available = check_ollama_server_reachable()

    # Determine overall status - we're "degraded" if Ollama isn't available
    status = "healthy" if ollama_available else "degraded"

    return {
        "status": status,
        "version": app.version,
        "ollama_status": "connected" if ollama_available else "unavailable",
        "ollama_url": get_settings().ollama_base_url,
    }


@app.post("/process-chunks/", response_model=ResponseModel)
async def process_chunks(
    file: UploadFile = File(...),
    llm_model: Optional[str] = Query(
        get_settings().llm_model,
        description="LLM model to use",
        # example=get_settings().llm_model,
    ),
    temperature: Optional[float] = Query(
        get_settings().temperature,
        description="Generation temperature",
        # example=get_settings().temperature,
    ),
    context_window: Optional[int] = Query(
        get_settings().context_window,
        description="LLM context window",
        # example=get_settings().context_window,
    ),
    custom_prompt: Optional[str] = Query(None, description="Custom prompt template"),
    n_qa_pairs: int = Query(
        get_settings().n_qa_pairs,
        description="Number of QA pairs to generate per chunk",
        ge=1,
        le=10,
    ),
    max_retries: int = Query(
        get_settings().default_max_retries,
        description="Maximum retry attempts for LLM failures (network, parsing, corrections)",
        ge=get_settings().min_retries,
        le=get_settings().max_retries_limit,
    ),
):
    """Process chunks from a JSON file and generate Q&A pairs for each chunk.

    This is the main endpoint of the API. It receives a JSON file containing
    text chunks, processes them in parallel, and returns the generated
    question-answer pairs along with metadata.

    Args:
        file (UploadFile): The JSON file containing the text chunks.
        llm_model (Optional[str], optional): The LLM model to use. Defaults to None.
        temperature (Optional[float], optional): The generation temperature.
                                                 Defaults to None.
        context_window (Optional[int], optional): The LLM context window size.
                                                  Defaults to None.
        custom_prompt (Optional[str], optional): A custom prompt template to use.
                                                 Defaults to None.
        num_pairs (int, optional): The number of Q&A pairs to generate per chunk.
                                   Defaults to 3.
        max_retries (int, optional): The maximum number of retry attempts.
                                     Defaults to 3.

    Raises:
        HTTPException: If the Ollama server is not reachable or the file is
                       not a valid JSON file.

    Returns:
        ResponseModel: An object containing the list of generated Q&A pairs
                       and processing metadata.

    """
    start_time = time.time()

    try:
        if not check_ollama_server_reachable():
            raise HTTPException(503, "Ollama server is not reachable.")

        if not file.filename.lower().endswith(".json"):
            raise HTTPException(
                400, f"Invalid file format: {file.filename}. Only JSON files accepted."
            )

        content = await file.read()
        try:
            data = json.loads(content)
            chunks = data.get("chunks", [])
            if not isinstance(chunks, list):
                raise HTTPException(
                    400, "Invalid JSON structure: 'chunks' must be a list."
                )
        except json.JSONDecodeError:
            raise HTTPException(400, "Invalid JSON format.")

        # Prepare processing parameters
        qa_id_counter = 1

        settings = get_settings()
        llm_model_to_use = llm_model or settings.llm_model
        temperature_to_use = (
            temperature if temperature is not None else settings.temperature
        )
        context_window_to_use = context_window or settings.context_window

        # Validate max_retries against settings limits
        max_retries = min(
            max(max_retries, settings.min_retries), settings.max_retries_limit
        )

        logger.info(
            f"Processing with LLM: {llm_model_to_use}, Temp: {temperature_to_use}, Context: {context_window_to_use}, "
            f"Num Pairs: {n_qa_pairs}, Max Retries: {max_retries}, Custom Prompt: {bool(custom_prompt)}"
        )

        llm_model_to_use = ensure_ollama_model(
            llm_model_to_use, fallback_model=settings.llm_model
        )

        # NUOVO: Initialize quality issues tracking
        quality_issues = {
            "empty_chunks": 0,
            "short_chunks": 0,
            "minimal_content": 0,
            "special_chars_only": 0,
            "no_output_generated": 0,
        }

        # NUOVO: Analyze chunks for quality issues
        for chunk in chunks:
            chunk_text = chunk.get("text", "").strip()

            if len(chunk_text) == 0:
                quality_issues["empty_chunks"] += 1
            elif len(chunk_text) < 10:
                quality_issues["short_chunks"] += 1
            elif len(chunk_text.split()) < 3:
                quality_issues["minimal_content"] += 1
            elif not any(c.isalpha() for c in chunk_text):
                quality_issues["special_chars_only"] += 1

        # Process chunks in parallel using ThreadPoolExecutor
        results = []
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=settings.max_workers) as executor:
            futures = {
                executor.submit(
                    generate_qa_from_chunk,
                    chunk=chunk.get("text", ""),
                    model=llm_model_to_use,
                    prompt=custom_prompt or QA_PROMPT_TEMPLATE,
                    temperature=temperature_to_use,
                    context_window=context_window_to_use,
                    chunk_id=chunk.get("id", -1),
                    num_pairs=n_qa_pairs,
                    max_retries=max_retries,  # Pass the parametric retry count
                ): chunk
                for chunk in chunks
            }

            for future in tqdm(futures, total=len(chunks), desc="Generating Q&A"):
                chunk_info = futures[future]
                try:
                    qa_result, metrics = future.result()

                    # NUOVO: Track no output issues
                    if not qa_result or not qa_result.strip():
                        quality_issues["no_output_generated"] += 1

                    results.append(
                        {
                            "id": chunk_info.get("id", -1),
                            "text": qa_result,
                            "metrics": metrics,
                            "token_count": get_token_count(chunk_info.get("text", "")),
                        }
                    )
                except Exception as e:
                    logger.error(
                        f"Error processing chunk {chunk_info.get('id', -1)}: {e}"
                    )

                    # NUOVO: Track processing failures
                    quality_issues["no_output_generated"] += 1

                    results.append(
                        {
                            "id": chunk_info.get("id", -1),
                            "text": "",
                            "metrics": {
                                "error": str(e),
                                "parsing_successful": False,
                                "quality_score": 0,
                            },
                            "token_count": get_token_count(chunk_info.get("text", "")),
                        }
                    )

        end_time = time.time()

        # Restructure the response
        final_chunks = []
        qa_id_counter = 1
        for result in results:
            if result["text"]:
                # Split the text into lines and process them in pairs (Q, A)
                pairs = result["text"].strip().split("\n")
                for i in range(0, len(pairs), 2):
                    if i + 1 < len(pairs):
                        question = pairs[i].strip()
                        answer = pairs[i + 1].strip()

                        # Create QAItem, removing potential prefixes like 'Q:' or 'A:' for cleanliness
                        # Using word boundary (\b) to prevent cutting off first letters of words like 'Alice'
                        clean_question = re.sub(
                            r"^(Q|Question)\b[:\s]*", "", question, flags=re.IGNORECASE
                        ).strip()
                        clean_answer = re.sub(
                            r"^(A|Answer)\b[:\s]*", "", answer, flags=re.IGNORECASE
                        ).strip()

                        # Handle empty questions or answers with unified correction strategy
                        if not clean_question or not clean_answer:
                            is_question_empty = not clean_question
                            correction_type = (
                                "missing_question"
                                if is_question_empty
                                else "missing_answer"
                            )
                            existing_context = (
                                clean_question if not is_question_empty else None
                            )

                            # Get the chunk_info for this result to use in retry
                            chunk_info = next(
                                (c for c in chunks if c.get("id") == result["id"]), None
                            )

                            logger.debug(
                                f"Found {correction_type} in chunk {result['id']}, using unified correction strategy"
                            )

                            # Apply the unified correction strategy with parametric retries
                            retry_question, retry_answer = generate_corrective_qa(
                                chunk=chunk_info.get("text", "") if chunk_info else "",
                                model=llm_model_to_use,
                                temperature=temperature_to_use,  # Use the same temperature for consistency
                                context_window=context_window_to_use,
                                chunk_id=result["id"],
                                correction_type=correction_type,
                                existing_question=existing_context,
                                max_retries=max_retries,  # Use the parametric max_retries from API param
                            )

                            # Update with corrected content if available
                            if retry_question:
                                clean_question = retry_question
                            if retry_answer:
                                clean_answer = retry_answer

                            # Skip this pair if we still can't generate a valid Q or A after retries
                            if not clean_question or not clean_answer:
                                logger.debug(
                                    f"Skipping QA pair with {correction_type} after {max_retries} retries"
                                )
                                # Track corrections in result metadata
                                if "corrections_applied" not in result:
                                    result["corrections_applied"] = 0
                                result["corrections_attempted"] = (
                                    result.get("corrections_attempted", 0) + 1
                                )
                                continue
                            else:
                                # Track successful corrections
                                if "corrections_applied" not in result:
                                    result["corrections_applied"] = 0
                                result["corrections_applied"] += 1
                                logger.debug(
                                    f"Successfully fixed {correction_type} in chunk {result['id']} with unified correction strategy"
                                )

                        # Calculate token count as sum of question and answer tokens
                        q_token_count = get_token_count(clean_question)
                        a_token_count = get_token_count(clean_answer)
                        qa_token_count = q_token_count + a_token_count

                        final_chunks.append(
                            QAItem(
                                text=f"{clean_question}\n{clean_answer}",
                                id=qa_id_counter,
                                token_count=qa_token_count,
                                id_source=result["id"],
                            )
                        )
                        qa_id_counter += 1

        total_successful_parses = sum(
            1 for r in results if r["metrics"].get("parsing_successful")
        )
        failed_parses = len(results) - total_successful_parses

        # NUOVO: Quality score onesto
        average_quality_score = calculate_honest_quality_score(
            chunks, results, quality_issues
        )

        metadata = Metadata(
            total_chunks=len(chunks),
            number_qa=n_qa_pairs,
            total_qa_pairs=len(final_chunks),
            failed_parses=failed_parses,
            average_quality_score=average_quality_score,
            quality_issues=quality_issues,
            llm_model=llm_model_to_use,
            temperature=temperature_to_use,
            context_window=context_window_to_use,
            custom_prompt_used=bool(custom_prompt and custom_prompt.strip()),
            source=file.filename,
            processing_time=round(end_time - start_time, 2),
        )

        # Log summary of generation results at the end
        successful_chunks = sum(
            1 for r in results if r["metrics"].get("parsing_successful")
        )
        total_generated_qa = len(final_chunks)
        expected_qa = len(chunks) * n_qa_pairs
        success_percentage = (
            (total_generated_qa / expected_qa) * 100 if expected_qa > 0 else 0
        )

        # Count corrections from result metadata
        total_corrections_applied = sum(
            r.get("corrections_applied", 0) for r in results
        )
        total_corrections_attempted = sum(
            r.get("corrections_attempted", 0) for r in results
        )
        corrections_success_rate = (
            (total_corrections_applied / total_corrections_attempted * 100)
            if total_corrections_attempted > 0
            else 0
        )

        logger.info(
            f"QA generation summary: {total_generated_qa}/{expected_qa} pairs generated ({success_percentage:.1f}%), "
            f"Successful chunks: {successful_chunks}/{len(chunks)}, "
            f"Failed chunks: {failed_parses}, "
            f"Corrections: {total_corrections_applied}/{total_corrections_attempted} successful ({corrections_success_rate:.1f}%), "
            f"Max retries: {max_retries}, Avg quality: {average_quality_score:.2f}"
        )

        return ResponseModel(chunks=final_chunks, metadata=metadata)

    except Exception as e:
        total_time = time.time() - start_time
        logger.error(
            f"Error processing chunks after {total_time:.2f} seconds: {e}",
            exc_info=True,
        )
        return JSONResponse(
            status_code=500,
            content={"error": f"An error occurred while processing chunks: {e}"},
        )
