import os
import time
import logging
from typing import List, Dict, Optional, Tuple

# Third-party imports (ensure these are installed: pip install langchain-huggingface faiss-cpu langchain-community)
try:
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
except ImportError as e:
    print(f"Error importing required libraries: {e}")
    print("Please install them using: pip install langchain-huggingface faiss-cpu langchain-community")
    exit(1) # Exit if essential libraries are missing


# --- Setup Logging ---
# Configure logging to write to a file named 'Logs/rag.log'
log_file = 'Logs/rag.log'
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO) # Set the logging level for this logger

# Prevent adding multiple handlers if this script is run multiple times in the same session
if not logger.handlers:
    # Create a file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO) # Set level for this handler

    # Create a formatter and set it for the handler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(file_handler)

    # Optional: Add a StreamHandler to also log to console during development/debugging
    # console_handler = logging.StreamHandler()
    # console_handler.setLevel(logging.INFO)
    # console_handler.setFormatter(formatter)
    # logger.addHandler(console_handler)

logger.info("--- Logging started ---") # Initial log message to confirm file setup


# --- Default Configuration (can be overridden during instantiation or method calls) ---
DEFAULT_K_RESULTS = 1
DEFAULT_SIMILARITY_THRESHOLD = 0.70
DEFAULT_INSTRUCTION = "As a cybersecurity expert, suggest an action to mitigate the threat described in the Input log."

class RAGSystem:
    """
    Encapsulates a Retrieval-Augmented Generation (RAG) system tailored for
    cybersecurity log analysis.

    This class loads a Hugging Face sentence embedding model and a FAISS vector
    store containing previously processed log summaries and suggested actions.
    It provides methods to:
    1. Search the vector store for log entries similar to a given query log.
    2. Generate formatted prompts for a large language model (LLM), optionally
       including the most relevant retrieved log and action as context.
    """

    def __init__(self,
                 embedding_model_name: str,
                 db_path: str,
                 allow_dangerous_deserialization: bool = True):
        """
        Initializes the RAG system by loading the embedding model and vector store.

        Args:
            embedding_model_name: The name or path of the Hugging Face sentence
                transformer model to use for generating embeddings (e.g.,
                'basel/ATTACK-BERT').
            db_path: The path to the directory containing the saved FAISS index
                (created using `FAISS.save_local`).
            allow_dangerous_deserialization: Whether to allow loading pickle files
                via FAISS's `load_local`.
                SECURITY WARNING: Only set this to True if you completely trust
                the source of the FAISS index files (`db_path`). Loading pickled
                data from untrusted sources can lead to arbitrary code execution.

        Raises:
            ValueError: If the embedding model cannot be initialized.
            FileNotFoundError: If the database path (`db_path`) does not exist or is not a directory.
            RuntimeError: If the vector store fails to load for other reasons.
            ImportError: If required libraries (langchain, faiss) are not installed.
        """
        self.embedding_model_name = embedding_model_name
        self.db_path = db_path
        self.allow_dangerous_deserialization = allow_dangerous_deserialization
        self.embeddings: Optional[HuggingFaceEmbeddings] = None
        self.vectorstore: Optional[FAISS] = None

        logger.info(f"Initializing RAGSystem with model: {self.embedding_model_name}, DB path: {self.db_path}")

        # 1. Initialize Embeddings
        try:
            logger.info(f"Loading embedding model: {self.embedding_model_name}...")
            # You can specify model_kwargs e.g., {'device': 'cuda'} if you have a GPU and PyTorch installed
            self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model_name)
            logger.info("Embedding model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize embedding model '{self.embedding_model_name}': {e}", exc_info=True)
            # Raise a specific error to indicate the failure point
            raise ValueError(f"Could not initialize embedding model: {e}")

        # 2. Load Vector Store (only if embeddings were loaded successfully)
        if self.embeddings:
            try:
                self.vectorstore = self._load_vector_db(
                    self.db_path,
                    self.embeddings,
                    self.allow_dangerous_deserialization
                )
                if self.vectorstore is None:
                    # The error was logged in _load_vector_db, raise exception to halt initialization
                    raise RuntimeError("Vector store failed to load. See logs for details.")
            except (FileNotFoundError, ModuleNotFoundError, RuntimeError, Exception) as e:
                # Catch potential errors from _load_vector_db (like FileNotFoundError, RuntimeError)
                # or other unexpected issues during the call.
                logger.error(f"An error occurred during vector store loading process: {e}", exc_info=True)
                raise # Re-raise the caught exception

        # Final check after attempts to load both components
        if self.embeddings is None or self.vectorstore is None:
            logger.critical("RAG System initialization failed. Embeddings or vector store could not be loaded.")
            # It's usually better to raise an error than continue with a non-functional system
            raise RuntimeError("RAG System could not be fully initialized. Check logs.")
        else:
            logger.info("RAG System initialized successfully.")

    def _load_vector_db(self, path: str, embeddings: HuggingFaceEmbeddings, allow_dangerous: bool) -> Optional[FAISS]:
        """
        Loads the FAISS index and associated metadata from a local directory.

        Args:
            path: The directory path where the FAISS index ('index.faiss')
                and metadata ('index.pkl') are stored.
            embeddings: The initialized HuggingFaceEmbeddings instance.
            allow_dangerous: Flag passed to FAISS.load_local to permit unsafe
                deserialization of the .pkl file.

        Returns:
            The loaded FAISS vector store instance if successful, otherwise None.

        Raises:
            FileNotFoundError: If the specified `path` does not exist or is not a valid directory.
            ModuleNotFoundError: If the 'faiss' library is not installed.
            RuntimeError: If loading fails for other reasons (e.g., corrupted files, permissions).
        """
        logger.info(f"Attempting to load vector store from directory: '{path}'...")
        if not os.path.isdir(path): # Check if it's a directory
            logger.error(f"Database path is not a valid directory: {path}")
            raise FileNotFoundError(f"Database path is not a valid directory: {path}")

        # Check for expected files (optional but good practice)
        index_file = os.path.join(path, "index.faiss")
        pkl_file = os.path.join(path, "index.pkl")
        if not os.path.exists(index_file) or not os.path.exists(pkl_file):
            logger.warning(f"Directory '{path}' does not contain the expected 'index.faiss' and 'index.pkl' files.")
            # FAISS.load_local might raise its own FileNotFoundError, but this warning is helpful.
            # Depending on strictness, could raise FileNotFoundError here directly.
            # raise FileNotFoundError(f"Required FAISS files not found in directory: {path}")

        try:
            vector_store = FAISS.load_local(
                folder_path=path, # Parameter name clarified
                embeddings=embeddings,
                allow_dangerous_deserialization=allow_dangerous,
                # Specify index_name if it's not 'index'
                # index_name="my_custom_index_name"
            )
            logger.info(f"Vector store loaded successfully from '{path}'.")
            return vector_store
        except ModuleNotFoundError as e:
            logger.error(f"Error loading vector store: {e}. FAISS library not found. Try 'pip install faiss-cpu' or 'pip install faiss-gpu'.", exc_info=True)
            raise # Re-raise ModuleNotFoundError
        except FileNotFoundError as e:
            # This might be raised by FAISS.load_local if files are missing/incorrectly named inside the dir
            logger.error(f"Error loading vector store from {path}: {e}. Ensure directory exists and contains valid 'index.faiss' and 'index.pkl'.", exc_info=True)
            raise # Re-raise FileNotFoundError
        except Exception as e:
            # Catch other potential issues like pickle errors, permission errors etc.
            logger.error(f"An unexpected error occurred while loading vector store from {path}: {e}", exc_info=True)
            # Wrap in a RuntimeError for clarity that loading failed
            raise RuntimeError(f"Failed to load vector store from {path}") from e

    def search_similar_logs(self,
                            query: str,
                            k: int = DEFAULT_K_RESULTS,
                            score_threshold: float = DEFAULT_SIMILARITY_THRESHOLD
                            ) -> List[Dict]:
        """
        Searches the vector store for log entries semantically similar to the query.

        It retrieves the top `k` most similar entries and then filters them based
        on the `score_threshold`. The score represents relevance, typically ranging
        from 0 (completely dissimilar) to 1 (identical), as normalized by
        `similarity_search_with_relevance_scores`.

        Args:
            query: The input text (e.g., a new log summary) to search against.
            k: The maximum number of similar documents to retrieve initially before filtering.
            score_threshold: The minimum relevance score (normalized, higher is better)
                required for a retrieved document to be considered a valid match.

        Returns:
            A list of dictionaries, where each dictionary represents a matched log
            meeting the score threshold. Each dictionary contains:
                - 'log_summary': The text content of the stored log summary.
                - 'suggested_action': The corresponding action stored in metadata.
                - 'score': The relevance score of the match (float).
            Returns an empty list if the vector store is not initialized, if no
            documents are found, or if none of the found documents meet the threshold.
        """
        if self.vectorstore is None:
            logger.error("Vector store is not available. Cannot perform similarity search.")
            return []

        # Log truncated query for brevity in logs, especially if queries can be very long
        truncated_query = query[:100] + "..." if len(query) > 100 else query
        logger.info(f"Searching vector store (k={k}, threshold={score_threshold}) for logs similar to: '{truncated_query}'")
        start_time = time.time()
        matches_above_threshold = [] # Renamed for clarity

        try:
            # Retrieve documents along with their relevance scores
            # Langchain's FAISS wrapper normalizes scores (higher is better)
            results_with_scores: List[Tuple[any, float]] = self.vectorstore.similarity_search_with_relevance_scores(
                query=query,
                k=k
                # score_threshold can also be passed directly here in some versions,
                # but filtering manually provides more logging control.
            )
            search_duration = time.time() - start_time
            logger.debug(f"Vector search completed in {search_duration:.4f} seconds.")

            if not results_with_scores:
                logger.info("No results returned from initial vector search.")
                return [] # Return empty list if search yields nothing

            logger.info(f"Retrieved {len(results_with_scores)} potential matches. Filtering by threshold >= {score_threshold}...")

            # Filter results based on the score threshold
            for doc, score in results_with_scores:
                truncated_content = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
                logger.debug(f"  - Potential Match | Score: {score:.4f} | Content: '{truncated_content}'")
                if score >= score_threshold:
                    # Ensure metadata exists and extract relevant fields safely
                    metadata = doc.metadata if hasattr(doc, 'metadata') else {}
                    # Fallback to page_content if 'log_summary' is missing in metadata
                    retrieved_summary = metadata.get('log_summary', doc.page_content)
                    # Fallback to 'N/A' if 'suggested_action' is missing
                    retrieved_action = metadata.get('suggested_action', 'N/A')

                    match_data = {
                        "log_summary": retrieved_summary,
                        "suggested_action": retrieved_action,
                        "score": float(score) # Ensure score is float
                    }
                    matches_above_threshold.append(match_data)

                    truncated_summary = retrieved_summary[:100] + "..." if len(retrieved_summary) > 100 else retrieved_summary
                    truncated_action = retrieved_action[:100] + "..." if len(retrieved_action) > 100 else retrieved_action
                    logger.info(f"  - Match found (Score: {score:.4f} >= {score_threshold}). Summary: '{truncated_summary}', Action: '{truncated_action}'")
                else:
                    # Log discarded matches only at DEBUG level if desired
                    logger.debug(f"  - Discarded match (Score: {score:.4f} < {score_threshold}).")

            if not matches_above_threshold:
                logger.info(f"No retrieved documents met the similarity threshold of {score_threshold}.")

            # Return the filtered list of matches
            return matches_above_threshold

        except Exception as e:
            logger.error(f"An error occurred during similarity search execution or processing: {e}", exc_info=True)
            return [] # Return empty list on error

    def get_prompt_with_rag(self,
                            input_context: str,
                            instruction: str = DEFAULT_INSTRUCTION,
                            k: int = DEFAULT_K_RESULTS,
                            score_threshold: float = DEFAULT_SIMILARITY_THRESHOLD
                            ) -> str:
        """
        Generates a formatted prompt for a language model, potentially enhancing
        it with context retrieved from the vector store based on the input_context.

        It first searches for similar logs using `search_similar_logs`. If one or
        more relevant logs (meeting the `score_threshold`) are found, the details
        (log summary and suggested action) of the *best* match (highest score)
        are included in the prompt as additional context. Otherwise, a standard
        prompt without retrieved context is generated.

        The prompt is formatted using the Alpaca style.

        Args:
            input_context: The primary input text for the LLM (e.g., a new log
                summary or event description needing analysis/action).
            instruction: The specific task instruction for the language model.
            k: The number of results to retrieve during the RAG context search.
            score_threshold: The minimum relevance score required for retrieved
                context to be included in the prompt.

        Returns:
            A formatted string prompt (Alpaca style) ready for use with an LLM.
            This prompt will include retrieved context if a relevant match was found,
            otherwise it will contain only the instruction and input context.
        """
        logger.info(f"Generating prompt for input context: '{input_context[:100]}...'")

        relevant_matches = [] # Initialize as empty list
        if self.vectorstore is None:
            logger.warning("Vector store not loaded. Cannot perform RAG search. Proceeding without retrieved context.")
        else:
            # Perform the search to find potential context
            relevant_matches = self.search_similar_logs(
                query=input_context,
                k=k,
                score_threshold=score_threshold
            )

        # Check if the search returned any matches that met the threshold
        if relevant_matches:
            # Use the first match (which should be the highest score based on FAISS search)
            # If search_similar_logs is modified to not sort, sorting might be needed here.
            best_match = relevant_matches[0]
            logger.info(f"Found relevant context (Best Score: {best_match['score']:.4f}). Including in prompt.")

            # Alpaca prompt format WITH retrieved context
            # Note: Improved formatting for clarity and potential parser compatibility
            alpaca_prompt_with_context = f"""Below is an instruction that describes a task, paired with an input that provides further context. Additionally, context from a similar past event is provided. Write a response that appropriately completes the request, taking the similar event context into consideration.

### Instruction:
{instruction}

### Input:
{input_context}

### Context (Similar Past Event):
Log Summary: {best_match['log_summary']}
Suggested Action: {best_match['suggested_action']}

### Response:
""" # The LLM generates text after this marker
            prompt = alpaca_prompt_with_context
            logger.debug(f"Generated prompt with context:\n{prompt}") # Log the full prompt at debug level

        else:
            # No relevant matches found OR vector store was unavailable
            if self.vectorstore: # Only log 'no relevant context found' if search was actually performed
                logger.info("No relevant context found meeting the threshold. Generating prompt without retrieved context.")
            # else: The warning about vector store unavailability was already logged.

            # Standard Alpaca prompt format WITHOUT context
            alpaca_prompt_no_context = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input_context}

### Response:
""" # The LLM generates text after this marker
            prompt = alpaca_prompt_no_context
            logger.debug(f"Generated prompt without context:\n{prompt}") # Log the full prompt at debug level

        return prompt.strip() # Return the final prompt, removing potential trailing whitespace

# --- End of File ---