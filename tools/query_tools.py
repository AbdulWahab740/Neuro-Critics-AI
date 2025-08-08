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
# Load the LLM and FAISS index
llm = load_llm()

embed = load_embedding_model()
faiss_index = load_faiss_index(embed)

@tool
def query_main_knowledge_base(query: str, topic_filter: Optional[str]) -> str:
    """
    Queries the main persistent FAISS knowledge base for general biomedical information.
    Optionally filter by 'topic_filter' if a specific domain like 'Parkinson_Disease' is relevant.
    Use this to get background knowledge, compare against existing research, or answer general questions.
    """
    if not faiss_index:
        return "Main knowledge base (FAISS) is not loaded or initialized."

    logging.info(f"Querying main knowledge base with: '{query}' (Topic filter: {topic_filter})")
    search_kwargs = {"k": 4}
    
    # Apply filter only if a topic is explicitly provided and exists in all_topics
    if topic_filter and topic_filter in core_topics().keys():
        search_kwargs["filter"] = {"topic": topic_filter}
        logging.info(f"Applying topic filter: {topic_filter}")
    else:
        logging.info("No valid topic filter provided or topic not found. Performing general search.")

    retriever = faiss_index.as_retriever(search_type="similarity", search_kwargs=search_kwargs)
    docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])
    logging.info(f"Retrieved {len(docs)} chunks from main knowledge base (FAISS).")

    local_llm_prompt = PromptTemplate.from_template(
        """Based on the following content from the knowledge base, summarize the relevant information to answer the user's query or provide background.
        Do not critique or generate the final answer yet, just extract and summarize the relevant facts.

        Context:
        {context}

        Query: {query}

        Summary from Knowledge Base:
        """
    )
    try:
        summary = (local_llm_prompt | llm | StrOutputParser()).invoke({"context": context, "query": query})
        return summary
    except Exception as e:
        logging.error(f"Error summarizing main KB content: {e}")
        traceback.print_exc()
        return f"Error summarizing content from main knowledge base: {e}"
