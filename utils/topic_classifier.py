import numpy as np
import logging
from config import TOPIC_CLASSIFICATION_THRESHOLD

def classify_topic(user_query: str, embedding_model, topic_embeddings) -> str:
    try:
        query_embedding = embedding_model.embed_query(user_query)
        similarities = {
            topic: np.dot(query_embedding, emb) / (np.linalg.norm(query_embedding) * np.linalg.norm(emb))
            for topic, emb in topic_embeddings.items()
        }

        most_similar = max(similarities, key=similarities.get)
        score = similarities[most_similar]
        if score >= TOPIC_CLASSIFICATION_THRESHOLD:
            return most_similar
        return None
    except Exception as e:
        logging.error(f"Error in topic classification: {e}")
        return None
