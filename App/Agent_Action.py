# Run Command: uvicorn Agent_Action:app --host 0.0.0.0 --port 8001
# Description: This script runs a FastAPI application that provides an API endpoint
#              to suggest mitigation actions for cybersecurity log summaries.
#              It utilizes a pre-trained language model (LLM) via Unsloth and
#              can optionally enhance prompts using a Retrieval-Augmented Generation (RAG)
#              system loaded from the 'rag_module'. Models and the RAG system
#              are initialized globally on startup.

import torch
import logging
from typing import Optional

# Third-party imports
try:
    from unsloth import FastLanguageModel
    from transformers import AutoTokenizer
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
except ImportError as e:
    # Use print here as logging might not be set up yet if imports fail early
    print(f"CRITICAL ERROR: Failed to import core libraries: {e}")
    print("Ensure transformers, unsloth, fastapi, uvicorn are installed.")
    exit(1)

# --- Setup Logging ---
log_file = 'Logs/agent_action.log'
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO) # Set default level

# Prevent adding multiple handlers if the script reloads (e.g., with uvicorn --reload)
if not logger.handlers:
    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO) # Set level for this handler

    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # Add handler to the logger
    logger.addHandler(file_handler)

    # Optional: Add console handler for simultaneous console output
    # console_handler = logging.StreamHandler()
    # console_handler.setLevel(logging.INFO)
    # console_handler.setFormatter(formatter)
    # logger.addHandler(console_handler)

logger.info("--- Starting Agent Action Service ---")


# --- Import the RAG System ---
from rag_module import RAGSystem, DEFAULT_INSTRUCTION as RAG_DEFAULT_INSTRUCTION
logger.info("Successfully imported RAGSystem from rag_module.")


# --- LLM Configuration ---
LLM_MODEL_ID = "Paul27/model_01" # Identifier for the Hugging Face model
MAX_SEQ_LENGTH = 2048           # Maximum sequence length for the LLM
DTYPE = None                    # Data type for model weights (None lets Unsloth choose)
LOAD_IN_4BIT = True             # Whether to load the model using 4-bit quantization


# --- RAG Configuration ---
RAG_EMBEDDING_MODEL = "basel/ATTACK-BERT" # Model used by RAG for embeddings
RAG_DB_PATH = 'feedback_faiss_db'         # Path to the FAISS vector database for RAG
RAG_ALLOW_DANGEROUS_DESERIALIZATION = True # Security flag for loading FAISS index (set True only if source is trusted)


# --- Global Variables ---
# These will hold the initialized models and systems once loaded on startup
model: Optional[FastLanguageModel] = None
tokenizer: Optional[AutoTokenizer] = None
rag_system_instance: Optional[RAGSystem] = None


# --- GPU Availability Check ---
gpu_available = torch.cuda.is_available()
if not gpu_available:
    logger.critical("CUDA GPU not available. Unsloth requires a GPU for optimal performance and loading.")
    # The application might still attempt to run but LLM loading will likely fail.
else:
    logger.info("CUDA GPU is available.")


# --- Load LLM Model (Global - Once on Startup) ---
# This section attempts to load the LLM and tokenizer if a GPU is available.
if gpu_available:
    logger.info(f"Attempting to load LLM model: {LLM_MODEL_ID}...")
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=LLM_MODEL_ID,
            max_seq_length=MAX_SEQ_LENGTH,
            dtype=DTYPE,
            load_in_4bit=LOAD_IN_4BIT,
            # token = "hf_...", # Add Hugging Face token if needed for private models
        )
        logger.info("LLM Model and Tokenizer loaded successfully.")

        logger.info("Optimizing LLM model for inference...")
        FastLanguageModel.for_inference(model) # Apply inference optimizations
        logger.info("LLM Model optimized successfully.")

    except Exception as e:
        logger.critical(f"Failed to load or optimize LLM model '{LLM_MODEL_ID}': {e}", exc_info=True)
        model = None # Ensure model is None if loading failed
        tokenizer = None
else:
    logger.warning("Skipping LLM model loading due to unavailable GPU.")


# --- Initialize RAG System (Global - Once on Startup) ---
# This section attempts to initialize the RAG system if the RAGSystem class was imported.
logger.info("Attempting to initialize RAG System globally...")
if RAGSystem: # Check if the class was successfully imported
    try:
        rag_system_instance = RAGSystem(
            embedding_model_name=RAG_EMBEDDING_MODEL,
            db_path=RAG_DB_PATH,
            allow_dangerous_deserialization=RAG_ALLOW_DANGEROUS_DESERIALIZATION
        )
        # Success message is logged internally by RAGSystem.__init__ if it succeeds
        # logger.info("RAG System initialized successfully.") # Redundant if RAGSystem logs success
    except (ValueError, FileNotFoundError, RuntimeError) as e:
        # Log specific, expected errors during RAG initialization
        logger.error(f"Failed to initialize RAG System: {e}", exc_info=True)
        rag_system_instance = None # Ensure instance is None on failure
    except Exception as e:
        # Log any other unexpected errors during RAG initialization
        logger.error(f"An unexpected error occurred during RAG System initialization: {e}", exc_info=True)
        rag_system_instance = None
else:
    logger.warning("RAGSystem class not available. Running without RAG capabilities.")


# --- FastAPI App Initialization ---
app = FastAPI(
    title="Cybersecurity Action Agent",
    description="API to suggest mitigation actions for log summaries, potentially enhanced with RAG context.",
    version="0.2.3", # Incremented version reflecting logging changes
)

# --- CORS Middleware ---
# Allow Cross-Origin Resource Sharing (CORS) for requests from any origin.
# Adjust origins list for stricter control in production environments.
origins = ["*"] # Allows all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Allows all HTTP methods
    allow_headers=["*"], # Allows all headers
)

# --- LLM Inference Function ---
def generate_llm_response(prompt: str, max_new_tokens: int = 150) -> str:
    """
    Performs inference using the globally pre-loaded Unsloth model and tokenizer.

    Args:
        prompt: The input prompt string for the language model.
        max_new_tokens: The maximum number of new tokens to generate.

    Returns:
        The generated text response from the LLM, or an error message string
        if inference fails or prerequisites (model, tokenizer, GPU) are missing.
    """
    # Pre-checks for necessary components
    if model is None or tokenizer is None:
        error_msg = "Error: LLM model or tokenizer not available for inference."
        logger.error(error_msg)
        return error_msg
    if not torch.cuda.is_available():
        error_msg = "Error: CUDA GPU not available for inference."
        logger.error(error_msg)
        return error_msg

    logger.info(f"Generating LLM response (max_new_tokens={max_new_tokens})...")
    try:
        # Prepare inputs for the model, moving them to the GPU
        inputs = tokenizer(
            [prompt],          # Input prompt(s) as a list
            return_tensors="pt" # Return PyTorch tensors
        ).to("cuda")           # Move tensors to the CUDA device

        # Generate response using the model
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens, # Limit the length of the generated response
            use_cache=True,                # Enable caching for faster generation (usually default)
            eos_token_id=tokenizer.eos_token_id, # End-of-sequence token ID
            # Set pad_token_id, falling back to eos_token_id if not explicitly set
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        )
        logger.info("LLM generation complete.")

        # Decode the generated tokens, skipping the input prompt tokens and special tokens
        # outputs[0] selects the first sequence in the batch (we only have one)
        # inputs["input_ids"].shape[1] gives the length of the input prompt tokens
        decoded_output = tokenizer.batch_decode(outputs[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]

        return decoded_output.strip() # Return the cleaned-up response

    except Exception as e:
        # Log any exception during the inference process
        error_message = f"Error during LLM inference: {e}"
        logger.error(error_message, exc_info=True)
        return error_message # Return the error message instead of generated text

# --- API Endpoints ---

@app.get("/")
async def read_root():
    """Provides a simple root endpoint indicating the API is running."""
    logger.info("Root endpoint '/' accessed.")
    return {"message": "Cybersecurity Action Agent API is running."}


@app.get("/get_action/")
async def get_log_action(log_summary: str):
    """
    Generates and retrieves a suggested mitigation action for a given log summary.

    This endpoint takes a log summary string as input.
    1. It attempts to generate an enhanced prompt using the RAG system (if available).
    2. If RAG fails or is unavailable, it falls back to a basic prompt format.
    3. It sends the final prompt to the LLM for action generation.
    4. It returns the generated action or raises an HTTPException on failure.

    Args:
        log_summary: The cybersecurity log summary text provided as a query parameter.

    Returns:
        A JSON response containing the suggested 'action'.

    Raises:
        HTTPException:
            - 400: If the 'log_summary' query parameter is empty.
            - 500: If the LLM fails to generate a response or an unexpected internal error occurs.
            - 503: Potentially if backend services (LLM/RAG models) are unavailable (currently handled as 500).
    """
    truncated_log = log_summary[:150] + "..." if len(log_summary) > 150 else log_summary
    logger.info(f"Received request for '/get_action/' with log summary: '{truncated_log}'")

    # Input validation
    if not log_summary:
        logger.warning("Received request with empty log summary.")
        raise HTTPException(
            status_code=400, # Bad Request
            detail="Log summary query parameter cannot be empty."
        )

    # --- Step 1: Get Prompt using RAG System (if available) ---
    final_prompt: Optional[str] = None # Initialize prompt variable
    if rag_system_instance:
        logger.info("Global RAG system instance available. Attempting to generate prompt with RAG.")
        try:
            # Call the RAG system's method to generate the prompt
            final_prompt = rag_system_instance.get_prompt_with_rag(
                input_context=log_summary,
                instruction=RAG_DEFAULT_INSTRUCTION # Use the potentially overridden default instruction
            )
            # RAGSystem's get_prompt_with_rag method should log details internally
            logger.info("Prompt successfully generated using RAG system.")
            # logger.debug(f"RAG Prompt:\n{final_prompt}") # Optionally log full prompt at DEBUG level
        except Exception as e:
            # Log error if RAG prompt generation fails
            logger.error(f"Error getting prompt from RAG system: {e}", exc_info=True)
            final_prompt = None # Ensure prompt is None to trigger fallback
            logger.warning("Falling back to basic prompt generation due to RAG error.")
    else:
        logger.info("Global RAG system instance not available. Using basic prompt.")


    # --- Step 2: Fallback to Basic Prompt if Needed ---
    if final_prompt is None:
        logger.info("Generating basic Alpaca prompt (RAG unavailable or failed).")
        # Standard Alpaca prompt format without retrieved context
        basic_alpaca_prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{RAG_DEFAULT_INSTRUCTION}

### Input:
{log_summary}

### Response:
""" # The LLM generates text after this marker
        final_prompt = basic_alpaca_prompt
        # logger.debug(f"Basic Prompt:\n{final_prompt}") # Optionally log full prompt at DEBUG level

    # --- Step 3: Generate Action using LLM ---
    logger.info("Sending final prompt to LLM for action generation.")
    try:
        # Call the inference function with the prepared prompt
        action_response = generate_llm_response(final_prompt)

        # Check if the inference function returned an error message
        if action_response.startswith("Error:"):
            logger.error(f"LLM generation failed: {action_response}")
            # Return a server error status code as the LLM backend failed
            raise HTTPException(
                status_code=500, # Internal Server Error (or 503 Service Unavailable)
                detail=f"Failed to generate action via LLM: {action_response}"
            )
        else:
            # LLM generation was successful
            truncated_action = action_response[:150] + "..." if len(action_response) > 150 else action_response
            logger.info(f"Successfully generated action: '{truncated_action}'")
            return {"action": action_response} # Return the successful response

    except HTTPException as http_exc:
        # Re-raise HTTPExceptions that were raised intentionally (e.g., from the LLM error check)
        raise http_exc
    except Exception as e:
        # Catch any other unexpected errors during the endpoint execution
        logger.critical(f"Unexpected error processing /get_action/ endpoint: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, # Internal Server Error
            detail=f"An unexpected internal error occurred while generating the action: {str(e)}"
        )

# --- End of File ---