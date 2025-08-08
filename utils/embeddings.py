import logging
import traceback
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from config import EMBEDDING_MODEL_NAME
from utils.data_fetch import core_topics

core_topic = core_topics()
@st.cache_resource
def load_embedding_model():
    try:
        embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            cache_folder='./models'
        )
        logging.info("Embedding model loaded successfully.")
        return embedding_model
    except Exception as e:
        logging.error(f"Failed to load embedding model: {e}")
        traceback.print_exc()
        st.error("Embedding model failed to load.")
        st.stop()

@st.cache_resource
def pre_embed_topics(_embedding_model_obj):
    """Pre-embeds topic names for classification."""
    logging.info("Pre-embedding topic names and aliases for classification...")
    topic_embeddings_dict = {}

    for topic_name, topic_info in core_topic.items():
        topic_description = topic_name.replace('_', ' ') + ". " + " ".join(topic_info.get("aliases", []))
        topic_embeddings_dict[topic_name] = _embedding_model_obj.embed_query(topic_description)
    logging.info(f"Pre-embedded {len(topic_embeddings_dict)} topics.")
    return topic_embeddings_dict
