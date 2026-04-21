import json
import os
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

# 1. Load the JSON data
file_path = "./support_chatbot_dataset_pretty.json"
with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

documents = []

# 2. Parse JSON into LangChain Documents
for entry in data:
    # We combine the conversation turns into a single text block for retrieval
    # This helps the AI find "Problem -> Solution" patterns
    conversation_text = ""
    for msg in entry["conversations"]:
        if msg["role"] != "system": # Skip system prompt to save space, or include if needed
            role_label = "Customer" if msg["role"] == "user" else "Assistant"
            conversation_text += f"{role_label}: {msg['content']}\n"
    
    # Create a document with metadata for better filtering later
    doc = Document(
        page_content=conversation_text.strip(),
        metadata={
            "intent": entry["intent"],
            "language": entry["language"],
            "has_email": entry["has_email"],
            "id": entry["id"]
        }
    )
    documents.append(doc)

print(f"Loaded {len(documents)} conversation documents.")

# 3. Generate Embeddings (Using the model from your main.ipynb)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# 4. Create and Save Vector Store
vector_store = FAISS.from_documents(documents, embeddings)
vector_store.save_local("faiss_support_index")

print("Vector store created and saved to 'faiss_support_index'")