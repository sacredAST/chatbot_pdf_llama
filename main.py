# Import necessary packages for FastAPI
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List

# Import packages for llama
from llama_index.core import SimpleDirectoryReader
from llama_index.core.readers.string_iterable import StringIterableReader
from llama_index.core import Settings, PromptTemplate
from llama_index.core import VectorStoreIndex
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Initialize the FastAPI app
app = FastAPI()

# Setup for llama embeddings using a HuggingFace model
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5", trust_remote_code=True)
Settings.embed_model = embed_model

# Setup for the Ollama language model
llm = Ollama(model="llama3", request_timeout=120.0)
Settings.llm = llm

# Define a Pydantic model for message
class Message(BaseModel):
    role: str  # Role of the message sender (e.g., user, system)
    content: str  # Content of the message

# Define a Pydantic model for the request containing a list of messages
class MessageRequest(BaseModel):
    messages: List[Message]

# Define the chat endpoint
@app.post("/chat")
async def chat(request: MessageRequest):
    texts = []
    
    # Process each message except the last one
    for message in request.messages[:-1]:
        if message.role == "user":
            texts.append(f"this is response of user: {message.content}")
        elif message.role == "system":
            texts.append(f"this is response of system: {message.content}")
        else:
            texts.append(f"this is response of you: {message.content}")
    
    # Load the processed texts into documents
    docs = StringIterableReader().load_data(
        texts=texts
    )
    
    # Create a vector store index from the documents
    index = VectorStoreIndex.from_documents(docs)
    
    # Create a query engine from the index
    query_engine = index.as_query_engine(streaming=True, similarity_top_k=4)
    
    # Query the engine with the content of the last message
    response = query_engine.query(request.messages[-1].content)
    
    # Return the response as a JSON object
    return JSONResponse(
        content={
            "content": response
        }
    )
