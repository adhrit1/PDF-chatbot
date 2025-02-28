from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from transformers import AutoTokenizer, AutoModelForCausalLM


def process_document(file_path):
    documents = SimpleDirectoryReader(input_files = [file_path]).load_data()
    index = VectorStoreIndex.from_documents(documents)
    return index

def rag_query(query: str, index):
    retrieved_docs = index.retrieve(query)
    
    # Generate response using the retrieved documents
    context = " ".join([doc.text for doc in retrieved_docs])
    response = generate_response(query, context)
    return response

def generate_response(query, context):
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
    model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")

    input_text = f"Context: {context}\n\nQuestion: {query}"
    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=100)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)