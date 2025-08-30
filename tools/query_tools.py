from langchain_community.tools import tool
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from utils.llm_loader import load_llm
from utils.faiss_utils import load_faiss_index
from utils.embeddings import load_embedding_model
import logging
import traceback
from utils.data_fetch import core_topics
from typing import Optional
from langchain_community.retrievers.bm25 import BM25Retriever
from langchain.retrievers import EnsembleRetriever

# Load the LLM and FAISS index
llm = load_llm()

embed = load_embedding_model()
faiss_index = load_faiss_index(embed)

@tool
def query_main_knowledge_base(
    query: str, 
    topic_filter: Optional[str] = None
) -> str:
    """
    Enhanced query function with confidence scoring and context optimization.
    """
    if not faiss_index:
        logging.info(f"Querying main knowledge base with: '{query}' (Topic filter: {topic_filter})")
    
    # Enhanced retrieval with more results for re-ranking
    search_kwargs = {"k": 10}  # Get more results for better selection
    if topic_filter and topic_filter in core_topics().keys():
        search_kwargs["filter"] = {"topic": topic_filter}

    # Get initial results from FAISS
    retriever = faiss_index.as_retriever(search_type="similarity", search_kwargs=search_kwargs)
    
    # Get documents from FAISS first to initialize BM25
    faiss_docs = retriever.invoke(query)
    
    # Initialize BM25 with the text content from FAISS results
    if faiss_docs:
        bm25_retriever = BM25Retriever.from_documents(faiss_docs)
        # Create ensemble with both retrievers
        ensemble_retriever = EnsembleRetriever(
            retrievers=[retriever, bm25_retriever],
            weights=[0.7, 0.3]
        )
        docs = ensemble_retriever.invoke(query)
    else:
        # Fallback to just FAISS if no documents found
        docs = faiss_docs
    
    # Generate response with confidence
    prompt = PromptTemplate.from_template("""
Based on the following context, provide a comprehensive and detailed answer to the query. 
Do not summarize or shorten unnecessarily. 
Include all relevant details, explanations, and context clearly. 
Preserve structure using headings, bullet points, and paragraphs.

Context:
{context}

Query: {query}

Answer:
""")

    try:
        # Generate response
        response = (prompt | llm | StrOutputParser()).invoke({
            "context": docs, 
            "query": query
        })
        logging.info(f"Generated response: {response}")
        return response
        
    except Exception as e:
        logging.error(f"Error in query_main_knowledge_base: {e}")
        traceback.print_exc()
        return f"Error processing your request: {str(e)}"

