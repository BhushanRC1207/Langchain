import json
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document  # Required for structured data
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

# 1. Setup LLM and Embeddings
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    temperature=0.3,
    max_new_tokens=512,
    huggingfacehub_api_token=os.getenv("HF_TOKEN")
)
model = ChatHuggingFace(llm=llm)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# 2. Data Ingestion (Replacing Mock Data with your JSON)
def load_support_dataset(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    docs = []
    for entry in data:
        # Combine conversation turns into one searchable context block
        conv_text = ""
        for msg in entry["conversations"]:
            if msg["role"] != "system":
                role = "User" if msg["role"] == "user" else "Assistant"
                conv_text += f"{role}: {msg['content']}\n"
        
        # Create a Document object
        docs.append(Document(page_content=conv_text.strip(), metadata={"intent": entry["intent"]}))
    return docs

# Load and Index
dataset_docs = load_support_dataset("./support_chatbot_dataset_pretty.json")
vector_store = FAISS.from_documents(dataset_docs, embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 3}) # Retrieve top 3 relevant cases

# 3. System Prompts (Enhanced with your specific Portal Features)
SYSTEM_PROMPT = """
You are an AI Support Assistant for the Ampcus Tech support portal. 
Use the provided conversation context to solve the user's issue.

Portal Features for Guidance:
- Security: Change password (min 8 chars, upper, lower, number, special char)
- Home Page: View All Tickets, Help Articles, FAQ, Create Ticket
- Priority Levels: High (4h), Medium (12h), Low (4h)

Rules:
1. Provide step-by-step guidance first.
2. If unresolved, collect: Full Name, Email ID, and Organisation.
3. Generate a professional support email addressed to support@ampcustech.com.
4. Support English and Hinglish.
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="history"),
    ("human", "Previous Cases/Context:\n{context}\n\nNew User Query: {question}"),
])

# 4. Chatbot Logic (Remains mostly the same)
class HelpdeskBot:
    def __init__(self):
        self.history = []
        self.chain = (
            {"context": retriever, "question": RunnablePassthrough(), "history": lambda x: self.history}
            | prompt
            | model
            | StrOutputParser()
        )

    def chat(self, user_input):
        response = self.chain.invoke(user_input)
        self.history.append(HumanMessage(content=user_input))
        self.history.append(AIMessage(content=response))
        return response

# 5. Execution
bot = HelpdeskBot()
print("AI Support: Welcome to Ampcus Tech Support. How can I help you? (Type 'quit')")

while True:
    user_query = input("User: ")
    if user_query.lower() in ['quit', 'exit']:
        break
    print(f"AI Support: {bot.chat(user_query)}")