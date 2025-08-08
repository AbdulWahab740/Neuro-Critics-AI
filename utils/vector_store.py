import json
import os
import logging
from langchain_community.vectorstores import FAISS # Changed from langchain_chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document # Import Document class

# --- IMPORT TOPICS FROM data_fetch.py ---
try:
    from utils.data_fetch import core_topics
    core_topic = core_topics()
    logging.info("Successfully imported core_topics from data_fetch.py")
except ImportError:
    logging.error("Error: Could not import core_topics from data_fetch.py. "
                  "Please ensure data_fetch.py exists and contains 'core_topics'.")
    exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
DATA_DIR = "./Data" 
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2' 
FAISS_INDEX_PATH = "./faiss_index" 
COLLECTION_NAME = "neuro_research_critics"
CHUNK_SIZE = 256
CHUNK_OVERLAP = 0

# --- Main function to build the vector store ---
def build_vector_store():
    logging.info("Starting vector store building process (FAISS)...")

    # Load Embedding Model using LangChain's HuggingFaceEmbeddings
    logging.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
    try:
        embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            cache_folder='./models'
        )
        logging.info("Embedding model loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load embedding model: {e}")
        return

    all_documents = []

    for topic_name, topic_info in core_topic.items():
        logging.info(f"\nProcessing articles for topic: {topic_name}")
        topic_jsonl_filename = f"{topic_name.lower()}_articles_data.jsonl"
        topic_jsonl_path = os.path.join(DATA_DIR, topic_name, topic_jsonl_filename)
        
        logging.info(f"Checking for file: {topic_jsonl_path}")
        if not os.path.exists(topic_jsonl_path):
            logging.warning(f"Warning: JSONL file not found for topic '{topic_name}' at '{topic_jsonl_path}'. Skipping.")
            continue

        processed_pmids_count = 0
        
        with open(topic_jsonl_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                try:
                    article_meta = json.loads(line)
                    pmid = article_meta.get("pmid")
                    
                    if not pmid:
                        logging.warning(f"Line {line_num+1}: Missing PMID. Skipping line.")
                        continue

                    title = article_meta.get("title", "").strip()
                    main_content_for_embedding = article_meta.get("content_for_embedding", "").strip()

                    if not main_content_for_embedding:
                        logging.debug(f"No 'content_for_embedding' found for PMID {pmid}. Skipping.")
                        continue

                    text_for_chunking = ""
                    if title:
                        text_for_chunking += title + ". "
                    text_for_chunking += main_content_for_embedding

                    if not text_for_chunking:
                        logging.debug(f"No combined text for chunking for PMID {pmid}. Skipping.")
                        continue
                    
                    # Prepare metadata for the Document object
                    metadata = {
                        "topic": topic_name,
                        "source_pmid": pmid,
                        "source_pmcid": article_meta.get("pmcid", "N/A"),
                        "article_title": title,
                        "article_abstract": article_meta.get("abstract", "N/A"),
                        "publication_year": article_meta.get("publication_year", "N/A"),
                        "journal": article_meta.get("journal", "N/A"), # Added journal metadata
                    }
                    
                    doc = Document(page_content=text_for_chunking, metadata=metadata)
                    all_documents.append(doc)
                    
                    processed_pmids_count += 1
                    if processed_pmids_count % 50 == 0:
                        logging.info(f"Processed {processed_pmids_count} articles for {topic_name}.")

                except json.JSONDecodeError as e:
                    logging.error(f"Error decoding JSON line {line_num+1} in '{topic_jsonl_path}': {e}. Line: {line.strip()}")
                except Exception as e:
                    logging.error(f"Unhandled error processing line {line_num+1} for topic {topic_name}: {e}")

        logging.info(f"Finished processing {processed_pmids_count} articles for topic '{topic_name}'.")

    logging.info(f"\nTotal documents prepared for indexing: {len(all_documents)}")

 
    logging.info(f"Creating FAISS index from {len(all_documents)} documents...")
    try:
        faiss_index = FAISS.from_documents(all_documents, embedding_model)
        logging.info("FAISS index created successfully.")
    except Exception as e:
        logging.error(f"Failed to create FAISS index: {e}")
        return

    # Save FAISS index locally
    logging.info(f"Saving FAISS index to {FAISS_INDEX_PATH}...")
    try:
        if not os.path.exists(FAISS_INDEX_PATH):
            os.makedirs(FAISS_INDEX_PATH)
        faiss_index.save_local(FAISS_INDEX_PATH)
        logging.info("FAISS index saved successfully.")
    except Exception as e:
        logging.error(f"Failed to save FAISS index: {e}")
        return

    logging.info("\nVector store creation complete!")

if __name__ == "__main__":
    build_vector_store()
