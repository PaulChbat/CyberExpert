# Run : python rag_db_creator.py

import orjson
import os
import time
from langchain_core.documents import Document # Use Langchain's Document class
# Removed: from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- Configuration ---
JSON_FILE_PATH = 'feedback.json'  # Your JSON file path
# Field containing the text to embed and search on
TEXT_FIELD_TO_EMBED = 'log_summary'
# Fields to store in metadata for retrieval
METADATA_FIELDS_TO_STORE = ['log_summary', 'suggested_action']

# Embedding model - paraphrase-multilingual-mpnet-base-v2 is good for multiple languages
EMBEDDING_MODEL_NAME = "basel/ATTACK-BERT"
# Output directory for the FAISS index
DB_SAVE_PATH = 'feedback_faiss_db'

# --- Code ---

def load_orjson_file(file_path):
    """Loads data from a JSON file using orjson."""
    print(f"Loading data from {file_path}...")
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return None
    try:
        with open(file_path, 'rb') as file:
            data = orjson.loads(file.read())
        if not isinstance(data, list):
             print(f"Error: Expected a JSON list, but got {type(data)}")
             return None
        print(f"Successfully loaded {len(data)} records.")
        return data
    except Exception as e:
        print(f"Error loading or parsing JSON file {file_path}: {e}")
        return None

def create_documents_from_data(data, text_field, metadata_fields):
    """Creates one Langchain Document per JSON item."""
    documents = []
    print("Creating one document per JSON item...")
    skipped_count = 0
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            print(f"Warning: Skipping item #{i} as it's not a dictionary.")
            skipped_count += 1
            continue

        page_content = item.get(text_field)
        if not page_content or not isinstance(page_content, str):
            print(f"Warning: Skipping item #{i} due to missing or invalid text field ('{text_field}').")
            skipped_count += 1
            continue

        metadata = {}
        essential_metadata_present = True
        for field in metadata_fields:
            if field in item:
                metadata[field] = item[field]
            else:
                 print(f"Warning: Metadata field '{field}' not found in item #{i}.")
                 if field in ['log_summary', 'suggested_action']:
                     essential_metadata_present = False
                 metadata[field] = None # Or decide how to handle missing optional metadata

        if not essential_metadata_present or not metadata.get('suggested_action'):
             print(f"Warning: Skipping item #{i} because essential field ('{text_field}' or 'suggested_action') is missing.")
             skipped_count +=1
             continue

        documents.append(Document(page_content=page_content.strip(), metadata=metadata))

    print(f"Created {len(documents)} documents.")
    if skipped_count > 0:
        print(f"Skipped {skipped_count} items due to missing data or wrong format.")
    return documents

# --- Main Execution ---
if __name__ == "__main__":
    print("--- Starting RAG DB Creation ---")

    # 1. Load Data
    start_time = time.time()
    loaded_data = load_orjson_file(JSON_FILE_PATH)
    if not loaded_data:
        exit(1)

    # 2. Create Documents (One per JSON item)
    langchain_docs = create_documents_from_data(
        loaded_data,
        TEXT_FIELD_TO_EMBED,
        METADATA_FIELDS_TO_STORE
    )
    if not langchain_docs:
        print("No documents were created. Exiting.")
        exit(1)

    # --- ADDED: Print the created documents (effectively the "chunks") ---
    print("\n--- Created Documents (Chunks) ---")
    # Limit printing if the list is very long, e.g., print first 5
    docs_to_print = langchain_docs #[:5] # Uncomment '[:5]' to limit printout
    if not docs_to_print:
         print("No documents to print.")
    else:
        for i, doc in enumerate(docs_to_print):
            print(f"Document {i}:")
            print(f"  Content (Summary): {doc.page_content}")
            print(f"  Metadata (Action, etc.): {doc.metadata}")
            print("-" * 30)
    # --- End of Added Print Section ---


    # 3. NO TEXT SPLITTING NEEDED - We use the documents directly

    # 4. Setup Embeddings
    print(f"\nLoading embedding model: {EMBEDDING_MODEL_NAME}...") # Added newline for spacing
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    print("Embedding model loaded.")

    # 5. Create and Save FAISS Vector Store
    print(f"Creating FAISS vector store from {len(langchain_docs)} documents...")
    try:
        vectorstore = FAISS.from_documents(langchain_docs, embeddings) # Use langchain_docs
        print(f"Vector store created with {vectorstore.index.ntotal} vectors.")
        print(f"Saving vector store to '{DB_SAVE_PATH}'...")
        vectorstore.save_local(DB_SAVE_PATH)
        end_time = time.time()
        print("--- RAG DB Creation Finished Successfully ---")
        print(f"Total time: {end_time - start_time:.2f} seconds.")
    except Exception as e:
        print(f"Error creating or saving FAISS vector store: {e}")
        print("--- RAG DB Creation Failed ---")
        exit(1)