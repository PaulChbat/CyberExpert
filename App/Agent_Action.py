# Run: uvicorn Agent_Action:app --host 0.0.0.0 --port 8001

import torch
from unsloth import FastLanguageModel
from transformers import AutoTokenizer
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
# Removed logging import
from typing import Optional

# --- Import the RAG System ---
try:
    from rag_db_search import RAGSystem, DEFAULT_INSTRUCTION as RAG_DEFAULT_INSTRUCTION
except ImportError:
    # Keep this print for critical failure feedback
    print("ERROR: Could not import RAGSystem from rag_module.py.")
    print("Ensure rag_module.py containing the RAGSystem class is in the same directory.")
    RAGSystem = None
    RAG_DEFAULT_INSTRUCTION = "As a cybersecurity expert, suggest an action to mitigate the threat"

# --- LLM Configuration ---
LLM_MODEL_ID = "Paul27/model_01"
MAX_SEQ_LENGTH = 2048
DTYPE = None
LOAD_IN_4BIT = True

# --- RAG Configuration ---
RAG_EMBEDDING_MODEL = "basel/ATTACK-BERT"
RAG_DB_PATH = 'feedback_faiss_db'
RAG_ALLOW_DANGEROUS_DESERIALIZATION = True

# --- Global Variables ---
model = None
tokenizer = None
rag_system_instance = None

# --- GPU Check ---
if not torch.cuda.is_available():
    # Keep this print for critical failure feedback
    print("CRITICAL ERROR: CUDA GPU not available. Unsloth requires a GPU.")
# else: # Removed info message
    # print("CUDA GPU is available.") # Optional: uncomment for basic feedback

# --- Load LLM Model (Global - Once on Startup) ---
try:
    if torch.cuda.is_available():
        # print(f"Loading LLM model: {LLM_MODEL_ID}...") # Optional basic feedback
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=LLM_MODEL_ID,
            max_seq_length=MAX_SEQ_LENGTH,
            dtype=DTYPE,
            load_in_4bit=LOAD_IN_4BIT,
            # token = "hf_...",
        )
        # print("LLM Model and Tokenizer loaded.") # Optional basic feedback

        # print("Optimizing LLM model for inference...") # Optional basic feedback
        FastLanguageModel.for_inference(model)
        # print("LLM Model optimized.") # Optional basic feedback
    # else: # Removed warning message
        # print("Skipping LLM model load due to unavailable GPU.")

except Exception as e:
    # Keep this print for critical failure feedback
    print(f"CRITICAL ERROR: Failed to load or optimize LLM model '{LLM_MODEL_ID}': {e}")
    model = None
    tokenizer = None

# --- Initialize RAG System (Global - Once on Startup) ---
# print("Initializing RAG System globally...") # Optional basic feedback
if RAGSystem:
    try:
        rag_system_instance = RAGSystem(
            embedding_model_name=RAG_EMBEDDING_MODEL,
            db_path=RAG_DB_PATH,
            allow_dangerous_deserialization=RAG_ALLOW_DANGEROUS_DESERIALIZATION
        )
        # print("RAG System initialized successfully.") # Optional basic feedback
    except (ValueError, FileNotFoundError, RuntimeError) as e:
        print(f"ERROR: Failed to initialize RAG System: {e}") # Keep print for failure
        rag_system_instance = None
    except Exception as e:
         print(f"ERROR: An unexpected error occurred during RAG System initialization: {e}") # Keep print for failure
         rag_system_instance = None
# else: # Removed warning message
    # print("RAGSystem class not available. Running without RAG capabilities.")


# --- FastAPI App Initialization ---
app = FastAPI(
    title="Cybersecurity Action Agent",
    description="API to suggest mitigation actions for log summaries, enhanced with RAG.",
    version="0.2.2", # Incremented version
)

# --- CORS Middleware ---
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- LLM Inference Function ---
def generate_llm_response(prompt: str, max_new_tokens: int = 150) -> str:
    """
    Performs inference using the pre-loaded Unsloth model.
    """
    if model is None or tokenizer is None:
         error_msg = "Error: LLM model or tokenizer not available."
         # print(error_msg) # Optional: print if needed, already returns error
         return error_msg
    if not torch.cuda.is_available():
         error_msg = "Error: CUDA GPU not available for inference."
         # print(error_msg) # Optional: print if needed, already returns error
         return error_msg

    try:
        inputs = tokenizer(
            [prompt],
            return_tensors="pt"
        ).to("cuda")

        # print(f"Generating LLM response (max_new_tokens={max_new_tokens})...") # Optional basic feedback
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            use_cache=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id
        )
        # print("LLM generation complete.") # Optional basic feedback

        decoded_output = tokenizer.batch_decode(outputs[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]

        return decoded_output.strip()

    except Exception as e:
        error_message = f"Error during LLM inference: {e}"
        # print(error_message) # Optional: print if needed, already returns error
        return error_message

# --- API Endpoints ---

@app.get("/")
async def read_root():
    """Returns a simple welcome message."""
    return {"message": "Cybersecurity Action Agent API with RAG (Global Init - No Logging)"}

# Updated get_action endpoint
@app.get("/get_action/")
async def get_log_action(log_summary: str):
    """
    Generates and retrieves a mitigation action for the provided log summary,
    using the globally initialized RAG system to potentially enhance context.
    """
    # print(f"Received request for action on log summary: '{log_summary[:150]}...'") # Optional basic feedback

    if not log_summary:
        # print("Received empty log summary.") # Optional basic feedback
        raise HTTPException(
            status_code=400,
            detail="Log summary query parameter cannot be empty."
        )

    # --- Step 1: Get Prompt using RAG System (if available) ---
    final_prompt = None
    if rag_system_instance:
        # print("Global RAG system instance available. Generating prompt with RAG.") # Optional basic feedback
        try:
            final_prompt = rag_system_instance.get_prompt_with_rag(
                input_context=log_summary,
                instruction=RAG_DEFAULT_INSTRUCTION
            )
            # print(f"Prompt generated by RAG:\n{final_prompt}") # Optional basic feedback
        except Exception as e:
            print(f"Error getting prompt from RAG system: {e}") # Keep print for failure feedback
            final_prompt = None
            # print("Falling back to basic prompt generation due to RAG error.") # Optional basic feedback
    # else: # Removed info message
        # print("Global RAG system instance not available. Will use basic prompt.")


    # If RAG wasn't available or failed, create a basic prompt
    if final_prompt is None:
         # print("Generating basic prompt.") # Optional basic feedback
         basic_alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:"""
         final_prompt = basic_alpaca_prompt.format(
             RAG_DEFAULT_INSTRUCTION,
             log_summary
         )
         # print(f"Basic prompt generated:\n{final_prompt}") # Optional basic feedback

    # --- Step 2: Generate Action using LLM ---
    # print("Sending prompt to LLM for action generation.") # Optional basic feedback
    try:
        action_response = generate_llm_response(final_prompt)

        if "Error:" in action_response:
            # print(f"LLM generation failed: {action_response}") # Optional basic feedback
            # The error message from generate_llm_response will be in the detail
            raise HTTPException(
                status_code=500,
                detail=f"Failed to generate action via LLM: {action_response}"
            )
        else:
            # print(f"Successfully generated action: '{action_response[:150]}...'") # Optional basic feedback
            return {"action": action_response}

    except HTTPException as http_exc:
        # Re-raise HTTPExceptions directly (like the one from LLM failure)
        raise http_exc
    except Exception as e:
        # Catch any other unexpected errors
        print(f"Unexpected error processing /get_action/ endpoint: {e}") # Keep print for failure feedback
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred while generating the action: {str(e)}"
        )

# --- End of File ---