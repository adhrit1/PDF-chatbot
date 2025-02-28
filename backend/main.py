from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)

load_dotenv()

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    file_path = f"uploads/{file.filename}"
    with open(file_path,"wb") as buffer:
        buffer.write(await file.read())
    return {"filename": file.filename,"Message":"File uploaded successfully"}

@app.get("/chat")
async def chat(query: str):
    # DO RAG
    return {"response":"Placeholder response"}


from llama_index.core import SimpleDirectoryReader, VectorStoreIndex

def process_document(file_path):
    documents = SimpleDirectoryReader(input_files = [file_path]).load_data()
    index = VectorStoreIndex.from_documents(documents)
    return index

from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")

def generate_response(query):
    inputs = tokenizer(query, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=50)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
