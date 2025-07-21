from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional, Dict, Tuple
from langchain_community.vectorstores import Qdrant
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from qdrant_client import QdrantClient
import os
import warnings
from urllib3.exceptions import NotOpenSSLWarning

warnings.filterwarnings("ignore", category=NotOpenSSLWarning)

from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────────────────────
# 🔍 Embeddings + Vector Store Setup
embeddings = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-large-en",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": False}
)

qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

vectorstore = Qdrant(
    collection_name="vector_db",
    client=qdrant_client,
    embeddings=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# ─────────────────────────────────────────────────────────────
# 💬 Prompt Template
prompt = PromptTemplate(
    input_variables=["chat_history", "context", "question"],
    template="""
You're a smart, helpful assistant trained on Bajaj Hindusthan Sugar Ltd.'s 92nd Annual Report (2023–24).

- ONLY answer using the information in the context.
- If the context lacks an answer, say: "I couldn't find that information in the report."
- Greet politely if user says hello.
- Don't repeat your intro unless greeted again.

---

📜 Chat history:
{chat_history}

📄 Context:
{context}

❓ Question:
{question}

💬 Answer:
"""
)

# ─────────────────────────────────────────────────────────────
# 🧠 LLM Setup
llm = ChatGroq(
    model="llama3-8b-8192",
    groq_api_key=os.getenv("GROQ_API_KEY"),
    max_tokens=512,
    temperature=0.2
)

llm_chain = LLMChain(llm=llm, prompt=prompt)
stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="context")

retrieval_chain = RetrievalQA(
    retriever=retriever,
    combine_documents_chain=stuff_chain,
    return_source_documents=True
)

# ─────────────────────────────────────────────────────────────
# 🗂️ Chat History Store
chat_histories: Dict[str, List[Tuple[str, str]]] = {}

# ─────────────────────────────────────────────────────────────
# 🚀 FastAPI Setup
app = FastAPI()

class QuestionRequest(BaseModel):
    question: str
    session_id: Optional[str] = "default_session"

@app.get("/")
def home():
    return {"message": "92nd Annual Report Assistant API is running."}

@app.post("/ask")
def ask_question(payload: QuestionRequest):
    session_id = payload.session_id or "default_session"

    if session_id not in chat_histories:
        chat_histories[session_id] = []

    # Fetch relevant documents
    docs = retriever.get_relevant_documents(payload.question)
    sources = [doc.page_content.strip()[:100] for doc in docs]
    print(sources)
    # Build chat history string for internal use
    chat_history_text = ""
    for q, a in chat_histories[session_id]:
        chat_history_text += f"User: {q}\nAssistant: {a}\n"

    # Build input for LLM
    inputs = {
        "chat_history": chat_history_text,
        "context": "\n\n".join([doc.page_content for doc in docs]),
        "question": payload.question
    }

    # Invoke LLM
    result = llm_chain.invoke(inputs)

    # Extract only the final answer
    if isinstance(result, dict):
        answer_text = result.get("text", "").strip()
    else:
        answer_text = str(result).strip()

    # Optionally update chat history (skip greetings)
    intro_phrases = [
        "i'm a helpful assistant", "i am a helpful assistant",
        "here to help", "legal assistant", "i assist with"
    ]
    if not any(p in answer_text.lower() for p in intro_phrases):
        chat_histories[session_id].append((payload.question, answer_text))

    # Final formatted response
    return {
        "question": payload.question,
        "answer": answer_text,
        "chat_history": chat_history_text.strip(),
        "sources": sources
    }

