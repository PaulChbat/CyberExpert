# Log Summary Agent API

## Overview

This file provides a simple FastAPI application designed to summarize cybersecurity logs. It utilizes the Groq API for fast LLM inference, integrated via a custom LangChain LLM wrapper. The API exposes endpoints to:

1.  Generate a concise, one-phrase summary for a given log entry.
2.  Submit simple feedback (the generated summary and a suggested action) which is stored locally in a JSON file.

The primary goal is to leverage the speed of the Groq API to quickly analyze and summarize potentially large volumes of log data.

## Features

* **Fast Log Summarization:** Uses Groq's LLM API (specifically configured for `meta-llama/llama-4-scout-17b-16e-instruct` in the code) to generate summaries.
* **LangChain Integration:** Implements a custom `GroqLLM` class that inherits from LangChain's `LLM` base class, making it compatible with the LangChain ecosystem (though usage here is direct).
* **FastAPI Backend:** Provides a robust and easy-to-use web API interface.
* **Simple Feedback Mechanism:** Allows users to submit feedback on summaries via an API endpoint, stored locally in `feedback.json`.
* **Environment Variable Configuration:** Requires the Groq API key to be set via the `GROQ_API_KEY` environment variable.
* **Basic Logging:** Logs application events and errors to `Logs/agent_summary.log`.
* **CORS Enabled:** Includes CORS middleware configured to allow requests from all origins (for development/simplicity).

## Prerequisites

* Python 3.8+
* pip (Python package installer)
* A Groq API Key

## Installation

1.  **Clone the repository (or download the script):**
    ```bash
    # If you have a git repository
    # git clone <your-repo-url>
    # cd <your-repo-directory>

    # Otherwise, just ensure you have the Agent_Summary.py file
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    Create a `requirements.txt` file with the following content:
    ```txt
    fastapi
    uvicorn[standard]
    langchain
    langchain-community
    pydantic
    groq
    python-dotenv # Optional, but helpful for managing .env files
    ```
    Then install them:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Create Logs Directory:**
    The application logs to the `Logs/` directory. Create it if it doesn't exist:
    ```bash
    mkdir Logs
    ```

## Configuration

The application requires your Groq API key.

* **Set the `GROQ_API_KEY` environment variable:**
    * **Option 1 (Recommended):** Create a `.env` file in the project root directory with the following content:
        ```
        GROQ_API_KEY=your_groq_api_key_here
        ```
        The script will automatically attempt to load this if `python-dotenv` is installed and handled appropriately (Note: The current script uses `os.getenv` directly, so you might need to load the `.env` file explicitly at the start or set the variable system-wide).
    * **Option 2:** Export the variable in your terminal session before running the application:
        ```bash
        # On macOS/Linux
        export GROQ_API_KEY='your_groq_api_key_here'

        # On Windows (Command Prompt)
        set GROQ_API_KEY=your_groq_api_key_here

        # On Windows (PowerShell)
        $env:GROQ_API_KEY='your_groq_api_key_here'
        ```

**Security Note:** Never hardcode your API keys directly into the script. Use environment variables or a secrets management system.

## Running the Application

Execute the following command in your terminal from the directory containing `Agent_Summary.py`:

```bash
uvicorn Agent_Summary:app --host 0.0.0.0 --port 8000 --reload
```

# Cybersecurity Action Agent API

## Overview

This code implements a FastAPI application that serves as a "Cybersecurity Action Agent". Its primary function is to receive a cybersecurity log summary and suggest a relevant mitigation action.

The core logic relies on a pre-trained Large Language Model (LLM) loaded using the Unsloth library for optimized inference on NVIDIA GPUs. Optionally, the agent can leverage a Retrieval-Augmented Generation (RAG) system (defined in `rag_module.py`) to retrieve relevant context from a vector database (FAISS) and enhance the prompt sent to the LLM, potentially leading to more accurate and context-aware action suggestions.

Models (LLM, Tokenizer, RAG system) are loaded into memory globally when the application starts to ensure faster response times for subsequent requests.

## Features

* **Action Suggestion API:** Provides an endpoint to get mitigation actions based on log summaries.
* **Optimized LLM Inference:** Utilizes Unsloth (`FastLanguageModel`) for efficient inference of transformer models on CUDA-enabled GPUs.
* **Quantization Support:** Configured to load the LLM using 4-bit quantization (`LOAD_IN_4BIT = True`) to reduce memory usage.
* **Optional RAG Enhancement:** Can integrate with a custom RAG system (`rag_module.py`) using a FAISS vector database to improve prompt context before LLM generation.
* **Global Model Loading:** Loads the LLM, tokenizer, and initializes the RAG system on application startup.
* **Configurable Components:** Model IDs, paths, and certain parameters are defined as constants within the script.
* **GPU Requirement:** Explicitly designed for and requires an NVIDIA GPU with CUDA support.
* **Basic Logging:** Logs service startup, requests, errors, and RAG/LLM operations to `Logs/agent_action.log`.
* **CORS Enabled:** Includes CORS middleware configured to allow requests from all origins.

## Prerequisites

* Python 3.8+
* pip (Python package installer)
* **NVIDIA GPU with CUDA Toolkit installed.** Unsloth relies heavily on CUDA for performance.
* Sufficient GPU VRAM to load the specified LLM (`Paul27/model_01` loaded in 4-bit).
* Potentially, `git` and `git-lfs` if required by Unsloth or model dependencies.

## Dependencies

This application requires several Python libraries and potentially a custom module:

* **Core Libraries:** `fastapi`, `uvicorn[standard]`, `torch`, `transformers`
* **Optimization:** `unsloth` (Requires specific PyTorch/CUDA versions - follow Unsloth installation guide carefully).
* **RAG Components (if used):**
    * `sentence-transformers` (Likely needed by `rag_module.py` for embeddings)
    * `faiss-cpu` or `faiss-gpu` (Depending on `rag_module.py` implementation and hardware)
* **Custom Module:** `rag_module.py` (This file must be present and contain the `RAGSystem` class implementation).
* **Optional:** `python-dotenv` (for managing environment variables if adapted)

## Installation

1.  **Clone the repository (or download the scripts):**
    Ensure you have both `Agent_Action.py` and the required `rag_module.py` in the same directory structure.

2.  **Create and activate a virtual environment (Recommended):**
    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install Dependencies:**
    * **Install Unsloth:** Follow the official Unsloth installation instructions corresponding to your specific CUDA version *very carefully*. This usually involves installing specific versions of `torch`, `transformers`, and `unsloth` itself. Example (check Unsloth repo for current commands):
        ```bash
        pip install "unsloth[cu121-newest-torch220] @ git+[https://github.com/unslothai/unsloth.git](https://github.com/unslothai/unsloth.git)"
        # Replace cu121 / torch220 with versions matching your system
        ```
    * **Install other dependencies:** Create a `requirements.txt` (excluding Unsloth if installed separately):
        ```txt
        fastapi
        uvicorn[standard]
        # torch, transformers (likely installed by unsloth)
        sentence-transformers
        faiss-cpu # or faiss-gpu
        python-dotenv
        ```
        Install them:
        ```bash
        pip install -r requirements.txt
        ```
    * **Hugging Face Login (Optional):** If the model (`LLM_MODEL_ID`) is private or gated, you might need to log in:
        ```bash
        pip install huggingface_hub
        huggingface-cli login
        ```

4.  **Create Logs Directory:**
    ```bash
    mkdir Logs
    ```

## Configuration

Several parameters are configured via constants directly within `Agent_Action.py`:

* `LLM_MODEL_ID`: Hugging Face identifier for the LLM (e.g., `"Paul27/model_01"`).
* `MAX_SEQ_LENGTH`: Max input sequence length for the LLM.
* `DTYPE`: Data type for model weights (None lets Unsloth decide).
* `LOAD_IN_4BIT`: Set to `True` to use 4-bit quantization.
* `RAG_EMBEDDING_MODEL`: Hugging Face identifier for the embedding model used by the RAG system (e.g., `"basel/ATTACK-BERT"`).
* `RAG_DB_PATH`: Path to the FAISS vector database file/directory (e.g., `'feedback_faiss_db'`).
* `RAG_ALLOW_DANGEROUS_DESERIALIZATION`: Security flag for loading FAISS index. **See RAG System Setup below.**

Currently, modifying these requires editing the script.

## RAG System Setup (Optional but Integrated)

The application is designed to work with a RAG system defined in `rag_module.py`.

1.  **`rag_module.py`:** This file *must* exist and contain a class named `RAGSystem`. This class should handle:
    * Loading the embedding model (`RAG_EMBEDDING_MODEL`).
    * Loading the FAISS index from `RAG_DB_PATH`.
    * Performing similarity searches against the index.
    * Formatting retrieved documents and the original input into a final prompt string suitable for the LLM.
    * It should expose a method like `get_prompt_with_rag(input_context: str, instruction: str) -> str`.

2.  **FAISS Database (`feedback_faiss_db`):**
    * This vector database is **not created** by the `Agent_Action.py` script.
    * You need a separate process/script to:
        * Gather the source documents (e.g., past log summaries and corresponding actions from `feedback.json` generated by `Agent_Summary`).
        * Generate embeddings for these documents using the `RAG_EMBEDDING_MODEL`.
        * Build and save a FAISS index containing these embeddings to the location specified by `RAG_DB_PATH`.

3.  **Security Warning (`RAG_ALLOW_DANGEROUS_DESERIALIZATION`):**
    * FAISS indices can sometimes be saved using Python's `pickle` mechanism. Loading pickled data from untrusted sources is a **major security risk**, as it can allow arbitrary code execution.
    * The `RAG_ALLOW_DANGEROUS_DESERIALIZATION` flag (if used by your `rag_module.py` when loading the index) should **only be set to `True` if you created the FAISS index yourself or fully trust its source.** Otherwise, keep it `False` and ensure your `rag_module.py` loads the index safely if possible.

If `rag_module.py` cannot be imported or `RAGSystem` fails to initialize (e.g., DB not found), the application will log a warning and fall back to using a basic prompt format without RAG enhancement.

## Running the Application

Ensure your GPU is available and `nvidia-smi` runs correctly. Execute the following command:

```bash
uvicorn Agent_Action:app --host 0.0.0.0 --port 8001 --reload
