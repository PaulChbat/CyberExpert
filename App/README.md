# Code Explanations for Log Summary and Action Agents

This document explains the functionality and structure of the Python code found in `Agent_Summary.py` and `Agent_Action.py`.

## Log Summary Agent (Agent_Summary.py)

### Overview

This file implements a FastAPI application using the Groq API via a custom LangChain wrapper to summarize cybersecurity logs. It provides API endpoints to generate summaries and collect basic feedback stored locally. The goal is fast log analysis using Groq.

### Features Implemented in Code

* **Fast Log Summarization:** Uses Groq's LLM API (specifically configured for `meta-llama/llama-4-scout-17b-16e-instruct` in the `setup_llm_instance` function) to generate summaries based on a specific prompt template.
* **LangChain Integration:** Contains a custom `GroqLLM` class inheriting from `langchain.llms.base.LLM`. This class wraps the `groq` client library to make calls to the Groq chat completion endpoint, adapting it to the LangChain interface. It uses `get_openai_callback` likely for token tracking, though its relevance for Groq may vary.
* **FastAPI Backend:** The script defines a FastAPI application (`app`) with associated endpoints.
* **Simple Feedback Mechanism:** Includes logic in the `/feedback/` endpoint to read from, append to, and write back to a local JSON file (`feedback.json`). Handles file existence and potential JSON decoding errors.
* **Environment Variable Configuration:** The `GroqLLM` class initializer explicitly requires a Groq API key, attempting to fetch it using `os.getenv("GROQ_API_KEY")`. A `ValueError` is raised if the key is not found.
* **Basic Logging:** Configures standard Python logging to output messages (INFO level by default) to `Logs/agent_summary.log`, including timestamps, logger name, level, and message. Prevents duplicate handlers on reload.
* **CORS Enabled:** Configures FastAPI CORS middleware (`CORSMiddleware`) to allow all origins (`*`), methods (`*`), and headers (`*`).

### Configuration within the Code

* **Groq API Key:** The `GroqLLM` class requires the Groq API key, retrieved via `os.getenv("GROQ_API_KEY")`.
* **Groq Model:** The LLM model name is hardcoded within the `setup_llm_instance` function (currently `"meta-llama/llama-4-scout-17b-16e-instruct"`). The temperature is also set here (0.1).
* **Feedback File:** The name of the local feedback file (`feedback.json`) is defined by the `SIMPLE_FEEDBACK_FILE` constant.
* **Log File:** The log file path (`Logs/agent_summary.log`) is defined near the logging setup.

### API Endpoints Functionality

* **`GET /`**: A root endpoint that returns a simple JSON message confirming the API is running. Logs access.
* **`GET /get_summary/`**:
    * Requires a `log` query parameter (string).
    * Validates that the `log` parameter is not empty (raises `HTTPException` 400 if it is).
    * Calls the `get_summary` function, which sets up the `GroqLLM` instance and prompt via `setup_llm_instance`.
    * Invokes the `GroqLLM` instance with the formatted prompt.
    * Returns a JSON response `{"summary": "..."}` on success.
    * Handles errors during summary generation, logging them and raising `HTTPException` 500 if `get_summary` returns `None` or an unexpected exception occurs.
* **`GET /feedback/`**:
    * Requires `log_summary` (string) and `suggested_action` (string) query parameters.
    * Safely reads existing data from `feedback.json`, handling file not found or invalid JSON errors by initializing an empty list.
    * Appends the new feedback (as a dictionary) to the list.
    * Writes the updated list back to `feedback.json` with indentation.
    * Returns a confirmation message on success.
    * Raises `HTTPException` 500 if writing to the file fails.

### Feedback Mechanism Logic

The `/feedback/` endpoint implements local file storage:
1.  Reads the `feedback.json` file.
2.  Handles cases where the file doesn't exist or contains invalid JSON by starting with an empty list.
3.  Appends a new dictionary `{"log_summary": ..., "suggested_action": ...}` to the list.
4.  Overwrites `feedback.json` with the updated list, formatted as JSON.

### Logging Setup

* Uses Python's built-in `logging` module.
* Configures a `FileHandler` to write to `Logs/agent_summary.log`.
* Sets the default logging level to `INFO`.
* Uses a specific `Formatter` for log message structure.
* Includes a check (`if not logger.handlers:`) to avoid adding multiple handlers if the script is reloaded (e.g., by uvicorn's `--reload` flag).

---

## Cybersecurity Action Agent (Agent_Action.py)

### Overview

This code implements a FastAPI application using the Unsloth library to run a local LLM for suggesting mitigation actions based on cybersecurity log summaries. It includes optional RAG (Retrieval-Augmented Generation) capabilities by integrating with a `rag_module.py` file and a FAISS vector database. Models are loaded globally on startup, requiring a CUDA-enabled GPU.

### Features Implemented in Code

* **Action Suggestion API:** Provides the `/get_action/` endpoint to generate mitigation actions.
* **Optimized LLM Inference:** Uses `unsloth.FastLanguageModel` to load and run a Hugging Face transformer model (`LLM_MODEL_ID`). Includes configuration for 4-bit quantization (`LOAD_IN_4BIT`) and optimizes the model for inference (`FastLanguageModel.for_inference(model)`).
* **GPU Requirement Logic:** Checks for CUDA availability using `torch.cuda.is_available()` and logs critical errors or warnings. LLM loading is skipped if no GPU is detected. The `generate_llm_response` function also includes checks for GPU availability before attempting inference.
* **Optional RAG Enhancement:**
    * Attempts to import `RAGSystem` from a local `rag_module.py`.
    * Initializes `RAGSystem` globally on startup if available, passing configuration constants (`RAG_EMBEDDING_MODEL`, `RAG_DB_PATH`, `RAG_ALLOW_DANGEROUS_DESERIALIZATION`).
    * The `/get_action/` endpoint logic attempts to call `rag_system_instance.get_prompt_with_rag()` first.
    * If RAG is unavailable or fails, it logs this and falls back to a basic Alpaca prompt format using the `log_summary` and `RAG_DEFAULT_INSTRUCTION`.
* **Global Model Loading:** The LLM model (`model`), tokenizer (`tokenizer`), and RAG system (`rag_system_instance`) are initialized as global variables outside the request lifecycle, intended to be loaded once on application startup. Error handling is included around the loading process.
* **Configurable Components:** Key settings (model IDs, paths, quantization, RAG settings) are defined as global constants within the script.
* **Basic Logging:** Configures logging to `Logs/agent_action.log` similarly to the Summary Agent, capturing service start, requests, RAG/LLM operations, GPU availability, and errors.
* **CORS Enabled:** Configures FastAPI CORS middleware for broad access (all origins, methods, headers).

### Configuration within the Code

* **LLM Settings:**
    * `LLM_MODEL_ID`: Hugging Face model identifier.
    * `MAX_SEQ_LENGTH`: Maximum sequence length for the tokenizer/model.
    * `DTYPE`: Data type for model weights (None lets Unsloth choose).
    * `LOAD_IN_4BIT`: Boolean flag for 4-bit quantization.
* **RAG Settings:**
    * `RAG_EMBEDDING_MODEL`: Identifier for the embedding model used by `rag_module.py`.
    * `RAG_DB_PATH`: Path to the FAISS vector database expected by `rag_module.py`.
    * `RAG_ALLOW_DANGEROUS_DESERIALIZATION`: Boolean flag potentially used by `rag_module.py` when loading the FAISS index (security consideration).
* **Log File:** The log file path (`Logs/agent_action.log`) is defined.

### RAG System Integration Logic

* The script attempts to import `RAGSystem` from `rag_module.py`.
* It initializes `rag_system_instance` globally, handling potential exceptions during initialization.
* In the `/get_action/` endpoint:
    * It checks if `rag_system_instance` was successfully initialized.
    * If yes, it calls `rag_system_instance.get_prompt_with_rag(log_summary, RAG_DEFAULT_INSTRUCTION)`.
    * If the RAG call fails or RAG wasn't initialized, it constructs a basic prompt string using the `log_summary` and `RAG_DEFAULT_INSTRUCTION` in a standard Alpaca format.
    * The resulting prompt (either from RAG or basic) is passed to `generate_llm_response`.

### API Endpoints Functionality

* **`GET /`**: Root endpoint returning a confirmation message. Logs access.
* **`GET /get_action/`**:
    * Requires a `log_summary` query parameter (string).
    * Validates that `log_summary` is not empty (raises `HTTPException` 400).
    * Attempts to generate a prompt using the RAG system via `rag_system_instance.get_prompt_with_rag()`.
    * If RAG fails or is unavailable, constructs a basic Alpaca-formatted prompt.
    * Calls the `generate_llm_response` function with the final prompt.
    * `generate_llm_response`:
        * Checks if `model`, `tokenizer`, and CUDA GPU are available. Returns an error string if not.
        * Tokenizes the prompt and moves tensors to the GPU.
        * Calls `model.generate()` with appropriate parameters (`input_ids`, `attention_mask`, `max_new_tokens`, etc.).
        * Decodes the generated tokens, skipping the prompt and special tokens.
        * Returns the cleaned response string or an error message if inference fails.
    * Checks if `generate_llm_response` returned an error string; if so, raises `HTTPException` 500.
    * If successful, returns a JSON response `{"action": "..."}`.
    * Catches other unexpected exceptions and raises `HTTPException` 500.

### Logging Setup

* Uses Python's `logging` module.
* Configures a `FileHandler` for `Logs/agent_action.log`.
* Sets the default level to `INFO`.
* Uses a standard `Formatter`.
* Includes check to prevent duplicate handlers on reload.
* Logs key events: startup, GPU status, model loading attempts/success/failure, RAG initialization attempts/failure, incoming requests, RAG usage/fallback, LLM calls, and errors.

### Important Notes on Code Behavior

* **GPU Dependency:** The core LLM functionality (`generate_llm_response`) explicitly requires a CUDA GPU and will return errors if one is not detected by `torch`. The model loading itself is also predicated on GPU availability.
* **Global State:** Relies on global variables (`model`, `tokenizer`, `rag_system_instance`) being successfully initialized at startup. Failures during initialization will prevent subsequent API calls from using those components.
* **RAG Module Reliance:** The RAG functionality is entirely dependent on the presence and correct implementation of `rag_module.py` and the availability of the specified FAISS database.
* **FAISS Deserialization Flag:** The `RAG_ALLOW_DANGEROUS_DESERIALIZATION` constant is passed to the `RAGSystem` initializer. The security implications depend on how `RAGSystem` uses this flag when loading the FAISS index.
* **Hardcoded Configuration:** Paths, model names, and other parameters are constants within the script, requiring code edits for changes.
