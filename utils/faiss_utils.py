from langchain_community.vectorstores import FAISS
import os
import logging
import traceback
import streamlit as st
from config import FAISS_INDEX_PATH

@st.cache_resource
def load_faiss_index(_embedding_model):
    """Loads and caches the FAISS vector store."""
    logging.info(f"Loading FAISS index from {FAISS_INDEX_PATH}...")
    try:
        if not os.path.exists(FAISS_INDEX_PATH):
            logging.error(f"FAISS index path '{FAISS_INDEX_PATH}' not found. Please run vector_store_builder.py first.")
            st.error(f"FAISS index not found at '{FAISS_INDEX_PATH}'. Please run 'vector_store_builder.py' first to build the index.")
            st.stop()

        faiss_index = FAISS.load_local(FAISS_INDEX_PATH, _embedding_model, allow_dangerous_deserialization=True)
        current_count = len(faiss_index.docstore._dict)
        logging.info(f"FAISS index loaded successfully. Total documents in index: {current_count}.")
        if current_count == 0:
            logging.warning("WARNING: FAISS index is empty. Please ensure 'vector_store_builder.py' was run successfully.")
            st.warning("FAISS index is empty. Retrieval will not work. Please run 'vector_store_builder.py' first!")
            st.stop()
        return faiss_index
    except Exception as e:
        logging.error(f"Failed to load FAISS index: {e}. Ensure index is built correctly.")
        traceback.print_exc()
        st.error(f"Failed to load FAISS index: {e}. Please ensure 'vector_store_builder.py' has been run to create the index.")
        st.stop()