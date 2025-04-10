# Run Command: uvicorn Agent_Summary:app --host 0.0.0.0 --port 8000
# Description: This script runs a FastAPI application designed to summarize
#              cybersecurity logs using the Groq API via a custom LangChain LLM wrapper.
#              It provides endpoints to get log summaries and submit simple feedback
#              which is stored locally in a JSON file.

import json
import os
import logging
from typing import Any, List, Optional, Dict

# Third-party imports
try:
    from langchain.prompts import PromptTemplate
    from pydantic import BaseModel, Field
    from langchain.llms.base import LLM
    from groq import Groq # Groq client library
    # Using get_openai_callback likely for general LLM call tracking/cost estimation within Langchain framework
    from langchain_community.callbacks.manager import get_openai_callback
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
except ImportError as e:
    # Use print as logging might not be configured yet
    print(f"CRITICAL ERROR: Failed to import necessary libraries: {e}")
    print("Ensure langchain, pydantic, groq, fastapi, uvicorn are installed.")
    exit(1)

# --- Setup Logging ---
log_file = 'Logs/agent_summary.log'
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO) # Set default level

# Prevent adding multiple handlers on script reload
if not logger.handlers:
    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # Add handler to the logger
    logger.addHandler(file_handler)

logger.info("--- Starting Summary Agent Service ---")

# --- Custom Groq LLM Wrapper Configuration ---
class GroqLLMConfig(BaseModel):
    """Pydantic model for configuring the GroqLLM wrapper."""
    model_name: str = Field(..., description="The name of the Groq model to use (e.g., 'llama3-8b-8192').")
    temperature: float = Field(0.0, description="Sampling temperature for generation (0.0 for deterministic).")
    groq_api_key: str = Field(..., description="The API key for accessing the Groq API.")

# --- Custom Groq LLM Wrapper for LangChain ---
class GroqLLM(LLM):
    """
    Custom LangChain LLM wrapper for interacting with the Groq API.

    This class adapts the Groq chat completion endpoint to the LangChain LLM interface.
    """
    config: GroqLLMConfig # Holds the configuration defined by GroqLLMConfig
    client: Any = None     # Holds the initialized Groq client instance

    def __init__(self, model_name: str, temperature: float = 0.0, groq_api_key: Optional[str] = None):
        """
        Initializes the GroqLLM wrapper.

        Args:
            model_name: The name of the Groq model to use.
            temperature: The sampling temperature. Defaults to 0.0.
            groq_api_key: The Groq API key. If None, attempts to read from
                          the "GROQ_API_KEY" environment variable.

        Raises:
            ValueError: If the Groq API key is not provided and not found in environment variables.
        """
        super().__init__() # Initialize the base LLM class
        # Retrieve API key from argument or environment variable
        resolved_api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        if not resolved_api_key:
            logger.critical("Groq API key must be provided directly or set as GROQ_API_KEY environment variable.")
            raise ValueError("Groq API key must be provided or set as GROQ_API_KEY environment variable.")
        else:
            logger.info("Groq API Key found.")

        # Store configuration using the Pydantic model
        self.config = GroqLLMConfig(
            model_name=model_name,
            temperature=temperature,
            groq_api_key=resolved_api_key # Store the resolved key
        )
        # Initialize the official Groq client
        self.client = Groq(api_key=self.config.groq_api_key)
        logger.info(f"GroqLLM initialized with model: {self.config.model_name}")

    # --- Properties for configuration access (optional but good practice) ---
    # Note: The original code had a slight confusion with property implementation.
    # The config is directly accessible via self.config. The properties below are
    # redundant if self.config is the primary way to access/set. Keeping structure
    # as provided but ideally simplified.

    @property
    def config(self) -> GroqLLMConfig:
        """Returns the current configuration."""
        # This implicitly uses the `self.config` instance variable defined in __init__
        return self._config

    @config.setter
    def config(self, value: GroqLLMConfig):
        """Sets the configuration."""
        self._config = value

    # --- Core LangChain LLM Methods ---
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """
        Makes a call to the Groq API chat completion endpoint.

        Args:
            prompt: The input prompt string.
            stop: Optional list of stop sequences (not directly used by Groq chat completion API,
                  but part of the standard LLM interface).

        Returns:
            The content of the message generated by the Groq model.

        Raises:
            Exception: If the API call fails.
        """
        logger.debug(f"Calling Groq API with model {self.config.model_name} and prompt: '{prompt[:100]}...'")
        try:
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.config.model_name,
                temperature=self.config.temperature,
            )
            generated_content = response.choices[0].message.content
            logger.debug("Groq API call successful.")
            return generated_content
        except Exception as e:
            logger.error(f"Error calling Groq API: {e}", exc_info=True)
            # Re-raise the exception to be handled by the caller
            raise

    @property
    def _llm_type(self) -> str:
        """Returns the identifier for this LLM type."""
        return "Groq"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Returns identifying parameters for logging and comparison."""
        return {"model_name": self.config.model_name, "temperature": self.config.temperature}

# --- Pydantic Model for Feedback Request Body ---
class FeedbackPayload(BaseModel):
    """Defines the expected structure for feedback data submitted via API."""
    log_summary: str
    suggested_action: str

# --- Constants ---
SIMPLE_FEEDBACK_FILE = "feedback.json" # File to store feedback locally

# --- Helper Functions ---
def setup_llm_instance(log: str) -> tuple[GroqLLM, str]:
    """
    Initializes the GroqLLM instance and formats the prompt for summarization.

    Args:
        log: The raw log string to be summarized.

    Returns:
        A tuple containing the initialized GroqLLM instance and the formatted prompt string.
    """
    logger.debug("Setting up GroqLLM instance and prompt.")
    # Initialize the custom Groq LLM wrapper
    # Note: Model name was hardcoded with potential options commented out
    llm = GroqLLM(
        model_name="meta-llama/llama-4-scout-17b-16e-instruct", 
        temperature=0.1,
        groq_api_key=os.getenv("GROQ_API_KEY") # API key must be set in environment
    )

    # Define the prompt template for log summarization
    template = """You are a cybersecurity expert. Your task is to analyze the provided log entry and generate a concise, one-phrase summary that captures the core event or finding. Return ONLY the summary phrase itself, without any additional explanation, labels, or introductory text.

Log:
{log}

Summary:""" # Instruction focuses on single-phrase output

    PROMPT = PromptTemplate(
        template=template, input_variables=["log"] # Define template and input variable
    )
    # Format the prompt with the actual log data
    prompt_text = PROMPT.format(log=log)
    logger.debug("LLM instance and prompt setup complete.")
    return llm, prompt_text

def get_summary(log: str) -> Optional[str]:
    """
    Generates a summary for the given log using the configured GroqLLM.

    Args:
        log: The log string to summarize.

    Returns:
        The generated summary string, or None if an error occurs during generation.
    """
    logger.info(f"Attempting to generate summary for log: '{log[:100]}...'")
    try:
        llm, prompt = setup_llm_instance(log)

        # Use get_openai_callback for potential tracking (though backend is Groq)
        # This context manager primarily tracks token counts and estimated costs
        # based on standard models, which might not be accurate for Groq.
        logger.debug("Entering get_openai_callback context for LLM call.")
        with get_openai_callback() as cb:
            summary = llm(prompt) # Make the call to the LLM (_call method)
            logger.debug(f"LLM call completed. Callback info: {cb}") # Log callback data (tokens, cost)

        logger.info("Summary generated successfully.")
        return summary
    except Exception as e:
        # Log errors occurring during LLM setup or the call itself
        logger.error(f"Failed to generate summary: {e}", exc_info=True)
        return None # Return None to indicate failure

# --- FastAPI Application Setup ---
app = FastAPI(
    title="Log Summary Agent",
    description="API using Groq and LangChain to summarize cybersecurity logs.",
    version="0.1.1", # Incremented version
)

# --- CORS Middleware ---
# Configure Cross-Origin Resource Sharing
origins = ["*"] # Allow all origins for simplicity (restrict in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Allow all standard methods
    allow_headers=["*"], # Allow all standard headers
)

# --- API Endpoints ---

@app.get("/")
async def read_root():
    """Provides a simple root endpoint to confirm the service is running."""
    logger.info("Root endpoint '/' accessed.")
    return {"message": "Log Summary Agent API is running."}


@app.get("/get_summary/")
async def get_log_summary(log: str):
    """
    Endpoint to generate and retrieve a summary for the provided log data.

    Accepts log data as a query parameter and returns its summary.

    Args:
        log (str): The log data string passed as a query parameter (e.g., /get_summary/?log=...).

    Returns:
        JSON response containing the generated 'summary'.

    Raises:
        HTTPException:
            - 400: If the 'log' query parameter is missing or empty.
            - 500: If summary generation fails or an unexpected error occurs.
    """
    logger.info(f"Received request for '/get_summary/'")
    # Input validation
    if not log:
        logger.warning("Request received with empty 'log' parameter.")
        raise HTTPException(
            status_code=400, # Bad Request
            detail="Required query parameter 'log' cannot be empty."
        )

    try:
        # Call the core summary generation function
        summary_response = get_summary(log) # This now returns Optional[str]

        # Check if the summary generation was successful
        if summary_response is not None:
            logger.info("Summary successfully generated and returned.")
            # Return the successful response
            return {"summary": summary_response}
        else:
            # Log the failure and raise an HTTPException if get_summary returned None
            logger.error("Summary generation failed (get_summary returned None).")
            raise HTTPException(
                status_code=500, # Internal Server Error
                detail="Failed to generate summary for the provided log. Check server logs for details."
            )
    except Exception as e:
        # Catch any other unexpected errors during the endpoint logic
        logger.critical(f"Unexpected error in /get_summary/ endpoint: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, # Internal Server Error
            detail=f"An unexpected server error occurred: {str(e)}"
        )


@app.get("/feedback/")
async def submit_feedback(log_summary: str, suggested_action: str):
    """
    Endpoint to receive and store feedback (log summary and suggested action).

    Appends the provided feedback to a local JSON file (`feedback.json`).
    Accepts feedback data as query parameters.

    Args:
        log_summary (str): The log summary provided as feedback.
        suggested_action (str): The suggested action provided as feedback.

    Returns:
        JSON response confirming successful receipt of feedback.

    Raises:
        HTTPException:
            - 500: If writing the feedback to the file fails.
    """
    logger.info(f"Received request for '/feedback/'")
    feedback_list = []

    # 1. Load existing feedback data safely
    try:
        if os.path.exists(SIMPLE_FEEDBACK_FILE):
            with open(SIMPLE_FEEDBACK_FILE, 'r', encoding='utf-8') as f:
                feedback_list = json.load(f)
                # Validate if the loaded data is a list
                if not isinstance(feedback_list, list):
                    logger.warning(f"File '{SIMPLE_FEEDBACK_FILE}' did not contain a list. Reinitializing.")
                    feedback_list = []
        else:
            logger.info(f"Feedback file '{SIMPLE_FEEDBACK_FILE}' not found. Will create a new one.")
            feedback_list = [] # Start fresh if file doesn't exist
    except json.JSONDecodeError:
        logger.warning(f"File '{SIMPLE_FEEDBACK_FILE}' contains invalid JSON. Reinitializing.")
        feedback_list = [] # Start fresh if file is corrupted
    except Exception as e:
        logger.error(f"Error reading feedback file '{SIMPLE_FEEDBACK_FILE}': {e}", exc_info=True)
        # Depending on requirements, might want to raise HTTPException here or just proceed with empty list
        feedback_list = []

    # 2. Prepare the new feedback entry
    new_feedback = {
        "log_summary": log_summary,
        "suggested_action": suggested_action
    }
    logger.debug(f"Prepared new feedback entry: {new_feedback}")

    # 3. Append new feedback to the list
    feedback_list.append(new_feedback)

    # 4. Save the updated list back to the file
    try:
        with open(SIMPLE_FEEDBACK_FILE, 'w', encoding='utf-8') as f:
            json.dump(feedback_list, f, indent=2) # Use indent for readability
        logger.info(f"Feedback successfully appended and saved to '{SIMPLE_FEEDBACK_FILE}'.")
    except Exception as e:
        logger.error(f"Failed to write feedback to file '{SIMPLE_FEEDBACK_FILE}': {e}", exc_info=True)
        # If saving fails, return a server error
        raise HTTPException(status_code=500, detail=f"Failed to save feedback: {str(e)}")

    return {"message": "Feedback received successfully"}

# --- End of File ---