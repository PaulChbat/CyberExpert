import os
import time
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import logging
from typing import List, Dict, Optional, Tuple

# --- Setup Logging ---
# Configure logging for use within a larger application
# The higher-level code can configure the root logger if needed.
logger = logging.getLogger(__name__)
# Set default logging level if not configured by higher-level code
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# --- Default Configuration (can be overridden during instantiation) ---
DEFAULT_K_RESULTS = 1
DEFAULT_SIMILARITY_THRESHOLD = 0.70
DEFAULT_INSTRUCTION = "As a cybersecurity expert, suggest an action to mitigate the threat described in the Input log."

class RAGSystem:
    """
    Encapsulates the functionality for a Retrieval-Augmented Generation system
    focused on cybersecurity log analysis.

    Loads embeddings and a vector store, and provides methods to search for
    similar logs and generate prompts incorporating retrieved context.
    """

    def __init__(self,
                 embedding_model_name: str,
                 db_path: str,
                 allow_dangerous_deserialization: bool = True):
        """
        Initializes the RAG system by loading embeddings and the vector store.

        Args:
            embedding_model_name: The name of the Hugging Face sentence transformer model to use.
            db_path: The path to the directory containing the saved FAISS index.
            allow_dangerous_deserialization: Whether to allow loading pickle files via FAISS.
                                             SECURITY WARNING: Only set to True if the db_path is trusted.

        Raises:
            ValueError: If the embedding model cannot be initialized.
            FileNotFoundError: If the database path does not exist.
            Exception: For other errors during FAISS loading.
        """
        self.embedding_model_name = embedding_model_name
        self.db_path = db_path
        self.allow_dangerous_deserialization = allow_dangerous_deserialization
        self.embeddings: Optional[HuggingFaceEmbeddings] = None
        self.vectorstore: Optional[FAISS] = None

        logger.info(f"Initializing RAGSystem with model: {self.embedding_model_name}, DB: {self.db_path}")

        # 1. Initialize Embeddings
        try:
            logger.info(f"Loading embedding model: {self.embedding_model_name}...")
            # Specify device if necessary, e.g., model_kwargs={'device': 'cuda'}
            self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model_name)
            logger.info("Embedding model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize embedding model '{self.embedding_model_name}': {e}", exc_info=True)
            # Raise an error or handle appropriately depending on desired system behavior
            raise ValueError(f"Could not initialize embedding model: {e}")

        # 2. Load Vector Store (only if embeddings loaded successfully)
        if self.embeddings:
            try:
                self.vectorstore = self._load_vector_db(
                    self.db_path,
                    self.embeddings,
                    self.allow_dangerous_deserialization
                )
                if self.vectorstore is None:
                     # Error logged in _load_vector_db, raise exception here
                     raise RuntimeError("Vector store failed to load.")
            except Exception as e:
                # Catch potential errors from _load_vector_db itself if needed
                logger.error(f"An unexpected error occurred during vector store loading: {e}", exc_info=True)
                raise

        if self.embeddings is None or self.vectorstore is None:
             logger.error("RAG System initialization failed.")
             # Depending on requirements, could raise an error here instead of continuing with None values
        else:
             logger.info("RAG System initialized successfully.")


    def _load_vector_db(self, path: str, embeddings: HuggingFaceEmbeddings, allow_dangerous: bool) -> Optional[FAISS]:
        """
        Loads the FAISS index and associated metadata from a local path.

        Args:
            path: The directory path where the FAISS index is stored.
            embeddings: The loaded embedding model instance.
            allow_dangerous: Flag to permit unsafe deserialization.

        Returns:
            The loaded FAISS vector store instance, or None if loading fails.
        """
        logger.info(f"Attempting to load vector store from '{path}'...")
        if not os.path.exists(path):
            logger.error(f"Database path not found: {path}")
            raise FileNotFoundError(f"Database path not found: {path}")

        try:
            vector_store = FAISS.load_local(
                path,
                embeddings,
                allow_dangerous_deserialization=allow_dangerous
            )
            logger.info("Vector store loaded successfully from %s.", path)
            return vector_store
        except ModuleNotFoundError as e:
             logger.error(f"Error loading vector store: {e}. Have you installed FAISS? Try 'pip install faiss-cpu' or 'pip install faiss-gpu'", exc_info=True)
             return None
        except Exception as e:
            logger.error(f"Error loading vector store from {path}: {e}", exc_info=True)
            return None

    def search_similar_logs(self,
                            query: str,
                            k: int = DEFAULT_K_RESULTS,
                            score_threshold: float = DEFAULT_SIMILARITY_THRESHOLD
                            ) -> List[Dict]:
        """
        Searches the vector store for log summaries similar to the query.

        Args:
            query: The input text (e.g., a log summary) to search for.
            k: The maximum number of results to retrieve initially.
            score_threshold: The minimum relevance score (0 to 1, higher is more similar)
                             required for a result to be considered a match.

        Returns:
            A list of dictionaries, where each dictionary represents a match
            meeting the threshold and contains 'log_summary', 'suggested_action',
            and 'score'. Returns an empty list if no matches meet the threshold
            or if the vector store is unavailable.
        """
        if self.vectorstore is None:
            logger.error("Vector store is not available for search.")
            return []

        logger.info(f"Searching (k={k}, threshold={score_threshold}) for logs similar to: '{query[:100]}...'") # Log truncated query
        start_time = time.time()
        matches = []
        try:
            # Retrieve results with relevance scores (normalized 0-1, higher is better)
            results_with_scores: List[Tuple[any, float]] = self.vectorstore.similarity_search_with_relevance_scores(
                query,
                k=k
            )
            end_time = time.time()
            logger.debug(f"Vector search took {end_time - start_time:.4f} seconds.")

            if not results_with_scores:
                logger.info("No results returned from vector search.")
                return []

            logger.info(f"Found {len(results_with_scores)} potential matches before thresholding:")
            for doc, score in results_with_scores:
                logger.debug(f"  - Potential Match Score: {score:.4f} | Content: {doc.page_content[:100]}...")
                if score >= score_threshold:
                    metadata = doc.metadata
                    retrieved_summary = metadata.get('log_summary', doc.page_content) # Use stored summary or fallback
                    retrieved_action = metadata.get('suggested_action', 'N/A') # Get action from metadata
                    matches.append({
                        "log_summary": retrieved_summary,
                        "suggested_action": retrieved_action,
                        "score": score
                    })
                    logger.info(f"  - Score {score:.4f} meets threshold. Adding match: Summary='{retrieved_summary[:100]}...', Action='{retrieved_action[:100]}...'")
                else:
                    logger.debug(f"  - Score {score:.4f} below threshold.")

            if not matches:
                 logger.info("No potential matches met the similarity threshold of %s.", score_threshold)

            return matches

        except Exception as e:
            logger.error(f"An error occurred during similarity search: {e}", exc_info=True)
            return []

    def get_prompt_with_rag(self,
                            input_context: str,
                            instruction: str = DEFAULT_INSTRUCTION,
                            k: int = DEFAULT_K_RESULTS,
                            score_threshold: float = DEFAULT_SIMILARITY_THRESHOLD
                            ) -> str:
        """
        Generates a formatted prompt for a language model, potentially including
        context retrieved from the vector store based on the input_context.

        Args:
            input_context: The primary input text (e.g., a log summary or event description).
            instruction: The instruction for the language model task.
            k: The number of results to retrieve for the RAG context search.
            score_threshold: The minimum score for retrieved context to be included.

        Returns:
            A formatted string prompt ready for use with a language model.
            If relevant context is found, it's included; otherwise, a standard prompt is returned.
        """
        if self.vectorstore is None:
            logger.warning("Vector store not loaded. Cannot perform RAG search. Proceeding without context.")
            matches = []
        else:
            # Perform the search using the instance's method
            matches = self.search_similar_logs(
                query=input_context,
                k=k,
                score_threshold=score_threshold
            )

        # Check if any matches were found *and* met the threshold
        if matches:
            # Use the first match (highest score meeting threshold)
            best_match = matches[0]
            logger.info(f"Found relevant context (Score: {best_match['score']:.4f}). Including in prompt.")

            # Alpaca prompt format including retrieved context
            alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. Additionally, an example context based on a similar past event is attached below, use it to inform your response.

### Instruction:
{}

### Input:
{}

### Context (Similar Past Event):
Log Summary: {}
Suggested Action: {}

### Response:"""
            prompt = alpaca_prompt.format(
                instruction,
                input_context,
                best_match['log_summary'],
                best_match['suggested_action']
                # The LLM will generate the text after "### Response:"
            )
        else:
            # No relevant matches found or vector store unavailable
            logger.info("No relevant context found or vector store unavailable. Generating prompt without context.")

            # Standard Alpaca prompt format without context
            alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:"""
            prompt = alpaca_prompt.format(
                instruction,
                input_context
                # The LLM will generate the text after "### Response:"
            )

        return prompt

# --- End of File ---