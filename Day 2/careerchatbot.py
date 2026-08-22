import os
import streamlit as st
import chromadb
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader

# Setup API key
os.environ["GROQ_API_KEY"] = "YOUR_API_KEY_HERE"

# LLM
llm = ChatGroq(
    temperature=0,
    model_name="openai/gpt-oss-20b"
)

# ChromaDB
client = chromadb.Client()
collection = client.get_or_create_collection("career_knowledge_base")


# PDF ingestion
def ingest_pdf(file):
    reader = PdfReader(file)
    text = ""

    for page in reader.pages:
        extracted_text = page.extract_text()

        if extracted_text:
            text += extracted_text + "\n"

    return text


# Streamlit UI
st.title("Career Guidance Chatbot")
st.markdown("Get personalized, grounded career advice using RAG + LLM.")


# Upload PDF
uploaded_file = st.file_uploader(
    "Upload a PDF document",
    type=["pdf"],
    accept_multiple_files=True
)


# Ingest documents
if uploaded_file:

    for file in uploaded_file:

        text = ingest_pdf(file)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=200,
            chunk_overlap=50
        )

        chunks = text_splitter.split_text(text)

        collection.add(
            documents=chunks,
            ids=[f"{file.name}_{i}" for i in range(len(chunks))]
        )

    st.success(f"{len(uploaded_file)} PDF(s) ingested successfully!")


# User query
user_query = st.text_input(
    "Ask a question about your career:"
)


# Get advice
if st.button("Get Advice") and user_query:

    # -----------------------------
    # 1. Vector Search
    # -----------------------------

    vector_results = collection.query(
        query_texts=user_query,
        n_results=5
    )

    vector_docs = vector_results["documents"][0]


    # -----------------------------
    # 2. Keyword Search
    # -----------------------------

    keywords = user_query.lower().split()

    keyword_docs = [
        doc
        for doc in vector_docs
        if any(k in doc.lower() for k in keywords)
    ]


    # -----------------------------
    # 3. Hybrid Search
    # -----------------------------

    hybrid_docs = list(set(vector_docs + keyword_docs))


    # -----------------------------
    # 4. Reranking
    # -----------------------------

    rerank_prompt = PromptTemplate.from_template(
        """
User Query:
{query}

Documents:
{docs}

Rank these documents from most relevant to least relevant
for providing career advice, including skills and companies.

Return the ranked list.
"""
    )

    rerank_chain = rerank_prompt | llm

    rerank_response = rerank_chain.invoke(
        {
            "query": user_query,
            "docs": hybrid_docs
        }
    )


    # -----------------------------
    # 5. Select Top Context
    # -----------------------------

    top_context = rerank_response.content.split("\n")[:3]


    # -----------------------------
    # 6. Final RAG Prompt
    # -----------------------------

    final_prompt = PromptTemplate.from_template(
        """
You are a career guidance AI assistant.

Based on the following resources:

{context}

Provide a personalized roadmap for the user:

- Skills to learn
- Recommend companies
- Steps to improve career readiness

User Query:
{query}
"""
    )


    # -----------------------------
    # 7. Generate Career Advice
    # -----------------------------

    rag_chain = final_prompt | llm

    career_advice = rag_chain.invoke(
        {
            "context": "\n".join(top_context),
            "query": user_query
        }
    )


    # -----------------------------
    # 8. Display Results
    # -----------------------------

    st.subheader("Top Retrieved Context")

    for doc in top_context:
        st.write(f"- {doc}")


    st.subheader("Personalized Career Advice")

    st.write(career_advice.content)