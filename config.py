import streamlit as st
import os
from dotenv import load_dotenv
load_dotenv()
from sentence_transformers import SentenceTransformer

# ✅ Let it download normally
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
FAISS_INDEX_PATH = "./faiss_index"
COLLECTION_NAME = "neuro_research_critics"
# GEMINI_API_KEY = st.secrets("GEMINI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "your_default_gemini_api_key_here")
TOPIC_CLASSIFICATION_THRESHOLD = 0.55
