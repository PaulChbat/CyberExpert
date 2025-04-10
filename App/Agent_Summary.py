# Run: uvicorn Agent_Summary:app --host 0.0.0.0 --port 8000

import json
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langchain.llms.base import LLM
from typing import Any, List, Optional, Dict
from groq import Groq
from langchain_community.callbacks.manager import get_openai_callback
from langchain.chains import RetrievalQA
import os
from fastapi import FastAPI, HTTPException
from typing import Optional 
from fastapi.middleware.cors import CORSMiddleware

class GroqLLMConfig(BaseModel):
    model_name: str = Field(..., description="The name of the Groq model to use.")
    temperature: float = Field(0.0, description="The temperature to use for sampling.")
    groq_api_key: str = Field(..., description="The API key for Groq.")

class GroqLLM(LLM):
    config: GroqLLMConfig
    client: Any = None

    def __init__(self, model_name: str, temperature: float = 0.0, groq_api_key: Optional[str] = None):
        super().__init__()
        groq_api_key = groq_api_key
        if not groq_api_key:
            raise ValueError("Groq API key must be provided or set as GROQ_API_KEY environment variable.")

        self.config = GroqLLMConfig(
            model_name=model_name,
            temperature=temperature,
            groq_api_key=groq_api_key
        )
        self.client = Groq(api_key=self.config.groq_api_key)

    @property
    def config(self) -> GroqLLMConfig:
        return self._config

    @config.setter
    def config(self, value: GroqLLMConfig):
        self._config = value

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response = self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self.config.model_name,
            temperature=self.config.temperature,
        )
        return response.choices[0].message.content

    @property
    def _llm_type(self) -> str:
        return "Groq"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {"model_name": self.config.model_name, "temperature": self.config.temperature}
    
# --- Pydantic Model for Feedback Request Body ---
class FeedbackPayload(BaseModel):
    log_summary: str
    suggested_action: str
# ---

def setup_llm_instance(log):
    llm = GroqLLM(
        model_name="meta-llama/llama-4-scout-17b-16e-instruct", #"llama3-8b-8192", 
        temperature=0.1,
        groq_api_key=os.getenv("GROQ_API_KEY")
    )
    # Define a custom prompt template
    template = """You are a cybersecurity expert, your job is to understand the given log and summarize it in a phrase. Return only the summary without adding anything else.
    Log:
    {log}

    Summary: """
    PROMPT = PromptTemplate(
        template=template, input_variables=["log"]
    )
    prompt = PROMPT.format(log=log)
    return llm,prompt

def get_summary(log):
    llm, prompt = setup_llm_instance(log)
    with get_openai_callback():
        answer = llm(prompt)
    return answer

# 1. Create a FastAPI instance
# Documentation will be available at http://127.0.0.1:8000/docs
# Alternative documentation at http://127.0.0.1:8000/redoc
app = FastAPI(
    title="Summary Agent",
    description="Python backend for Log summarization.",
    version="0.1.0",
)

# --- Example Endpoints ---

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,  # Usually needed if you ever deal with cookies/auth headers
    allow_methods=["*"],     # Allows GET, POST, OPTIONS, etc.
    allow_headers=["*"],     # Allows common headers
)

# 2. Define a root endpoint (GET request)
@app.get("/")
async def read_root():
    """
    Returns a simple welcome message.
    """
    return {"message": "Python backend for Log summarization"}

# 3. Define an endpoint with a path parameter (GET request)
#    It also includes an optional query parameter 'q'
@app.get("/get_summary/")
async def get_log_summary(log: str):
    """
    Generates and retrieves a summary for the provided log data.

    - **log**: The log data string to be summarized (passed as a query parameter).
    """
    # Input validation (optional but good practice)
    if not log:
        raise HTTPException(
            status_code=400,  # Bad Request
            detail="Log parameter cannot be empty."
        )

    try:
        # Call the actual summary generation function
        summary_response = get_summary(log)

        # Check if the summary generation was successful
        if summary_response:
            # Return the successful response (FastAPI converts dict/str to JSON)
            return {"summary": summary_response}
        else:
            # Raise an HTTPException if get_summary indicated failure (e.g., returned None/False)
            raise HTTPException(
                status_code=500,  # Internal Server Error (or choose a more specific code if applicable)
                detail="Failed to generate summary for the provided log."
            )
    except Exception as e:
        # Catch potential unexpected errors during summary generation
        # Log the error for debugging (important for production)
        print(f"ERROR: Unexpected error in get_summary: {e}") # Replace with proper logging
        raise HTTPException(
            status_code=500, # Internal Server Error
            detail=f"An unexpected error occurred while generating the summary: {e}"
        )
    
@app.get("/feedback/")
async def submit_feedback(log_summary: str, suggested_action: str):
    feedback_list = []
    SIMPLE_FEEDBACK_FILE = "feedback.json"
    # 1. Try to load existing data (simplified error handling)
    try:
        with open(SIMPLE_FEEDBACK_FILE, 'r', encoding='utf-8') as f:
            feedback_list = json.load(f)
            # Basic check if it's a list
            if not isinstance(feedback_list, list):
                print(f"Warning: File '{SIMPLE_FEEDBACK_FILE}' didn't contain a list. Starting fresh.")
                feedback_list = []
    except FileNotFoundError:
        # File doesn't exist yet, starting with an empty list is fine
        pass
    except json.JSONDecodeError:
        # File exists but is not valid JSON, start fresh
        print(f"Warning: File '{SIMPLE_FEEDBACK_FILE}' contains invalid JSON. Starting fresh.")
        feedback_list = []

    # 2. Prepare the new feedback entry
    new_feedback = {
        "log_summary": log_summary,
        "suggested_action": suggested_action
    }

    # 3. Append new feedback
    feedback_list.append(new_feedback)

    # 4. Save the updated list back to the file (simplified error handling)
    try:
        with open(SIMPLE_FEEDBACK_FILE, 'w', encoding='utf-8') as f:
            json.dump(feedback_list, f, indent=2) # Using indent=2 for readability
    except Exception as e:
        # If saving fails, return an server error
        raise HTTPException(status_code=500, detail=f"Failed to write to feedback file: {e}")

    return {"message": "Feedback received successfully"}
