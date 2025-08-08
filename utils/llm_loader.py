from langchain_google_genai import ChatGoogleGenerativeAI
from config import GEMINI_API_KEY
import logging, traceback
import streamlit as st

@st.cache_resource
def load_llm():
    """Loads and caches the LLM for use in the application."""
    if not GEMINI_API_KEY:
        st.error("GEMINI_API_KEY not set.")
        st.stop()
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.5,
            max_tokens=2000,
            api_key=GEMINI_API_KEY
        )
        return llm
    except Exception as e:
        traceback.print_exc()
        st.error("LLM failed to initialize.")
        st.stop()
