import os
import logging
import uuid
import json
import traceback
from langchain_community.tools import tool
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyMuPDFLoader
from utils.embeddings import load_embedding_model
from utils.llm_loader import load_llm
from langchain.text_splitter import RecursiveCharacterTextSplitter
from utils.embeddings import load_embedding_model
from langchain_community.retrievers.bm25 import BM25Retriever
from langchain.retrievers import EnsembleRetriever

llm = load_llm()
embeddings = load_embedding_model()
# --- Text Splitter for PDF ingestion ---
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=64,
    separators=["\n\n", "\n", " ", ""]
)
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
import streamlit as st

# Tool 1: Load PDF and create a temporary retriever
@tool
def load_pdf_and_create_temp_retriever(pdf_path: str) -> str:
    """
    Loads a PDF file from a given path, processes its content,
    and creates a temporary in-memory FAISS retriever for it.
    Returns a confirmation message.
    """
    logging.info(f"Attempting to load PDF from: {pdf_path}")
    if not os.path.exists(pdf_path):
        return f"Error: PDF file not found at {pdf_path}"

    try:
        loader = PyMuPDFLoader(pdf_path)
        docs = loader.load()

        processed_docs = []
        pdf_uuid = str(uuid.uuid4())
        for i, doc in enumerate(docs):
            doc.metadata["source_pdf_id"] = pdf_uuid
            doc.metadata["page_number"] = doc.metadata.get("page", i)
            doc.metadata["source_type"] = "uploaded_pdf"
            processed_docs.append(doc)

        chunks = text_splitter.split_documents(processed_docs)
        logging.info(f"Loaded {len(docs)} pages from PDF. Split into {len(chunks)} chunks.")

        # Create an in-memory FAISS index for the temporary PDF
        temp_faiss_vectorstore = FAISS.from_documents(
            documents=chunks,
            embedding=embeddings, # Use the global embedding model
        )
        st.session_state.temp_pdf_retriever = temp_faiss_vectorstore.as_retriever(search_kwargs={"k": 10})
        st.session_state.temp_pdf_docs = chunks

        return f"Successfully loaded and processed PDF. Content is ready for critique or query. PDF ID: {pdf_uuid}"
    except Exception as e:
        logging.error(f"Error loading or processing PDF '{pdf_path}': {e}")
        traceback.print_exc()
        return f"Error processing PDF: {e}. Please check the file."


# Tool 2: Query the temporary PDF retriever
@tool
def query_temp_pdf_for_critique(query: str, source_pdf_id: str = None) -> str:
    """
    Queries the content of the currently loaded temporary PDF for critical analysis.
    Use this to get specific information or passages from the uploaded PDF to critique.
    'source_pdf_id' is optional, but if multiple PDFs are loaded in a session, it can help
    specify which one, though currently only one temp PDF is active.
    """
    if st.session_state.temp_pdf_retriever:
        logging.info(f"Querying temporary PDF for critique with: '{query}'")
        bm25_retriever = BM25Retriever.from_documents(st.session_state.temp_pdf_docs)
        retriever = st.session_state.temp_pdf_retriever
        
        ensemble_retriever = EnsembleRetriever(retrievers=[retriever, bm25_retriever], weights=[0.7, 0.3])
        st.session_state.temp_pdf_retriever = ensemble_retriever
        docs = st.session_state.temp_pdf_retriever.invoke(query)
        context = "\n\n".join([doc.page_content for doc in docs])
        logging.info(f"Retrieved {len(docs)} chunks from temporary PDF.")

        local_llm_prompt = PromptTemplate.from_template(
            """Based on the following content from the uploaded paper, summarize the relevant information to answer the user's specific query for critical analysis.
            Do not critique yet, just extract and summarize the relevant facts.

            Context:
            {context}

            Query: {query}

            Summary from PDF:
            """
        )
        try:
            summary = (local_llm_prompt | llm | StrOutputParser()).invoke({"context": context, "query": query})
            return summary
        except Exception as e:
            logging.error(f"Error summarizing temp PDF content: {e}")
            traceback.print_exc()
            return f"Error summarizing content from temporary PDF: {e}"
    else:
        return "No PDF has been loaded or processed for temporary querying."

