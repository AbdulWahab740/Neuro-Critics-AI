import streamlit as st
import os
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import tool
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
import logging
import uuid
import tempfile
import traceback
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
FAISS_INDEX_PATH = "./faiss_index"
COLLECTION_NAME = "neuro_research_critics" 
GEMINI_API_KEY = st.secrets("GEMINI_API_KEY")
TOPIC_CLASSIFICATION_THRESHOLD = 0.55

try:
    from data_fetch import core_topics
    all_topics = core_topics()
    logging.info("Successfully imported all_topics from data_fetch.py")
except ImportError:
    st.error("Error: Could not import all_topics from data_fetch.py. "
             "Please ensure data_fetch.py exists and contains 'core_topics'.")
    st.stop()

# Debug: Print API key status
logging.info(f"DEBUG: GEMINI_API_KEY loaded: {bool(GEMINI_API_KEY)} (Length: {len(GEMINI_API_KEY) if GEMINI_API_KEY else 0})")

# --- Cache resources to avoid re-loading on every rerun ---

@st.cache_resource
def load_embedding_model():
    """Loads and caches the HuggingFace embedding model."""
    logging.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
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
        st.error(f"Failed to load embedding model: {e}. Please check your internet connection or model name.")
        st.stop()

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

@st.cache_resource
def load_llm():
    """Initializes and caches the Google Gemini LLM."""
    logging.info(f"Initializing LLM: gemini-2.0-flash...")
    if not GEMINI_API_KEY:
        st.error("GEMINI_API_KEY is not set. Please set it in your environment variables or Streamlit secrets.")
        st.stop()
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.5,
            max_tokens=2000,
            timeout=None,
            max_retries=2,
            api_key=GEMINI_API_KEY,
        )
        _ = llm.invoke("Hello, are you working?")
        logging.info("LLM initialized and tested successfully.")
        return llm
    except Exception as e:
        logging.error(f"Failed to initialize LLM: {e}. Check API key and model availability.")
        traceback.print_exc()
        st.error(f"Failed to initialize LLM: {e}. Please check your GEMINI_API_KEY and model access.")
        st.stop()

@st.cache_resource
def pre_embed_topics(_embedding_model_obj):
    """Pre-embeds topic names for classification."""
    logging.info("Pre-embedding topic names and aliases for classification...")
    topic_embeddings_dict = {}
    for topic_name, topic_info in all_topics.items():
        topic_description = topic_name.replace('_', ' ') + ". " + " ".join(topic_info.get("aliases", []))
        topic_embeddings_dict[topic_name] = _embedding_model_obj.embed_query(topic_description)
    logging.info(f"Pre-embedded {len(topic_embeddings_dict)} topics.")
    return topic_embeddings_dict

logging.info("Attempting to load embedding model, FAISS index, and LLM...")
embedding_model = load_embedding_model()
faiss_index = load_faiss_index(embedding_model)
llm = load_llm()
topic_embeddings = pre_embed_topics(embedding_model)
logging.info("All core components (embedding, FAISS, LLM, topic embeddings) loaded.")

# --- Text Splitter for PDF ingestion ---
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    separators=["\n\n", "\n", " ", ""]
)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "temp_pdf_docs" not in st.session_state:
    st.session_state.temp_pdf_docs = []
if "temp_pdf_retriever" not in st.session_state:
    st.session_state.temp_pdf_retriever = None
if "last_uploaded_filename_processed" not in st.session_state:
    st.session_state.last_uploaded_filename_processed = ""

# --- Custom Tools for our Agent ---

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
            embedding=embedding_model, # Use the global embedding model
        )
        st.session_state.temp_pdf_retriever = temp_faiss_vectorstore.as_retriever(search_kwargs={"k": 4})
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

# Tool 3: Query the main FAISS knowledge base
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
    if topic_filter and topic_filter in all_topics:
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


# Tool 4: Store processed PDF content into the main FAISS knowledge base
@tool
def store_pdf_to_main_knowledge_base() -> str:
    """
    Stores the content of the last processed PDF (from load_pdf_and_create_temp_retriever)
    into the main persistent FAISS knowledge base.
    This action should be called after a PDF has been processed and potentially critiqued.
    """
    if not st.session_state.temp_pdf_docs:
        return "No PDF content available to store. Please load a PDF first."
    if not faiss_index:
        return "Main knowledge base (FAISS) is not initialized. Cannot store PDF."

    logging.info(f"Attempting to store {len(st.session_state.temp_pdf_docs)} chunks from temporary PDF to main FAISS knowledge base.")
    try:
        first_chunk_content = st.session_state.temp_pdf_docs[0].page_content
        inferred_topic = classify_topic(first_chunk_content) # Use the classification logic

        docs_to_store = []
        for doc in st.session_state.temp_pdf_docs:
            # Ensure metadata is compatible with FAISS filtering if needed later
            # FAISS metadata is typically flat.
            new_metadata = doc.metadata.copy()
            new_metadata["source_type"] = "uploaded_pdf_persisted"
            if inferred_topic:
                new_metadata["topic"] = inferred_topic # Add inferred topic to metadata
            
            # Create a new Document object with updated metadata
            docs_to_store.append(Document(page_content=doc.page_content, metadata=new_metadata))

        # Add documents to the FAISS index
        faiss_index.add_documents(docs_to_store)
        
        # Save the updated FAISS index to disk
        faiss_index.save_local(FAISS_INDEX_PATH)

        logging.info("PDF content successfully stored in main FAISS knowledge base.")
        return f"PDF content (ID: {st.session_state.temp_pdf_docs[0].metadata.get('source_pdf_id', 'N/A')}) successfully added to the main knowledge base. Inferred topic: {inferred_topic if inferred_topic else 'None'}"
    except Exception as e:
        logging.error(f"Error storing PDF content to main FAISS knowledge base: {e}")
        traceback.print_exc()
        return f"Error storing PDF content: {e}"

# --- Topic Classification Function (uses global embedding_model and topic_embeddings) ---
def classify_topic(user_query: str) -> str:
    """
    Classifies the user query into one of the predefined topics using semantic similarity.
    Returns the most confident topic name or None if below threshold.
    """
    if not embedding_model or not topic_embeddings:
        logging.warning("Embedding model or topic embeddings not initialized for classification.")
        return None

    try:
        query_embedding = embedding_model.embed_query(user_query)
        similarities = {}
        for topic_name, topic_emb in topic_embeddings.items():
            similarity = np.dot(query_embedding, topic_emb) / (np.linalg.norm(query_embedding) * np.linalg.norm(topic_emb))
            similarities[topic_name] = similarity

        if similarities:
            most_similar_topic = max(similarities, key=similarities.get)
            max_similarity = similarities[most_similar_topic]
            if max_similarity >= TOPIC_CLASSIFICATION_THRESHOLD:
                logging.info(f"Classified query to topic: '{most_similar_topic}' with similarity {max_similarity:.4f}")
                return most_similar_topic
            else:
                logging.info(f"Query similarity ({max_similarity:.4f}) below threshold ({TOPIC_CLASSIFICATION_THRESHOLD}) for any topic.")
                return None
        return None
    except Exception as e:
        logging.error(f"Error during topic classification: {e}")
        traceback.print_exc()
        return None

# --- LangChain Agent Setup ---
@st.cache_resource
def setup_agent():
    tools = [
        load_pdf_and_create_temp_retriever,
        query_temp_pdf_for_critique,
        query_main_knowledge_base,
        store_pdf_to_main_knowledge_base
    ]
    logging.info(f"Defined {len(tools)} tools for the agent.")

    # Modified agent_prompt to include MessagesPlaceholder for chat history
    agent_prompt = PromptTemplate.from_template(
        """You are an intelligent AI assistant capable of analyzing biomedical research,
        loading new PDF documents, querying a knowledge base, and performing critical reviews.
        You have access to the following tools:

        Use the tools to address the user's request.
        If the 'Current PDF Status' indicates a PDF is already loaded:
        - If the user's query is explicitly about the uploaded PDF (e.g., "Critique this paper", "Summarize the findings of the PDF", "What is the methodology in this document?"), then prioritize using `query_temp_pdf_for_critique`.
        - If the user explicitly asks to store the uploaded PDF (e.g., "Store this paper", "Save this PDF"), use `store_pdf_to_main_knowledge_base`.
        - **If the user's query is a general question and NOT explicitly about the loaded PDF (e.g., "What is Parkinson's disease?", "Tell me about epilepsy treatments"), then use `query_main_knowledge_base`. these are some filter topics so you can take one of that from the relevancy with the query : ['Alzheimer_Disease', 'Stroke_Management', 'Epilepsy', 'Parkinson', 'Diagnostic', 'Neurotransmitter', 'Sclerosis', 'Migraine', 'Neurodevelopmental_disorder', 'TBI', 'Amyotrophic_Lateral_Sclerosis', 'Neuroinflammation', 'Sleep_disorder', 'Brain', 'Social_neurology'] **
        - **DO NOT ask for a PDF path again if one is already loaded.**

        When a user uploads a PDF, your first action should be to use the `load_pdf_and_create_temp_retriever` tool with the path to the uploaded PDF file.
        Finally, synthesize all information to generate a comprehensive critical review or answer.

        Your response should be comprehensive, well-structured, and insightful.

        Current PDF Status: {current_pdf_status} # New placeholder for PDF status

        Previous conversation:
        {chat_history} # Placeholder for chat history

        Answer the following question:

        {input}

        {agent_scratchpad}
        """
    )
    try:
        # Create the agent with the updated prompt
        agent = create_tool_calling_agent(llm, tools, agent_prompt)
        # Pass the chat history to the agent executor
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
        logging.info("Agent Executor initialized successfully.")
        return agent_executor
    except Exception as e:
        logging.error(f"Failed to setup agent: {e}")
        traceback.print_exc()
        st.error(f"Failed to setup agent: {e}. Check LLM and tool definitions.")
        st.stop()

# Initialize the agent executor
agent_executor = setup_agent()
logging.info("Agent setup complete. Starting Streamlit UI rendering...")

# --- Streamlit UI ---
st.set_page_config(page_title="🔬 Neuro Critics RAG & Critique Agent (FAISS)", page_icon="🔬")

st.title("🔬 Neuro Critics RAG & Critique Agent (FAISS)")
st.markdown(
    """
    Ask questions about Parkinson's Disease, Epilepsy, Sclerosis, Migraine, ALS, TBI or Alzheimer's Disease..<br>
    It also contian Diagnositics, Brain, Stroke, sleep disorder and Neurotransmitter related infomrations.<br>
    **OR** upload a PDF paper and ask me to 'Critique this paper' or ask specific questions about it. <br>
   
    """,
    unsafe_allow_html=True
)

with st.sidebar:
    st.header("Upload PDF for Critique")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf", accept_multiple_files=False, key="pdf_uploader")

    if uploaded_file is not None and uploaded_file.name != st.session_state.last_uploaded_filename_processed:
        st.info("Processing uploaded PDF... Please wait.")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_pdf_path = tmp_file.name
        
        try:
            chat_history_for_agent = []
            for msg in st.session_state.messages:
                if msg["role"] == "user":
                    chat_history_for_agent.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    chat_history_for_agent.append(AIMessage(content=msg["content"]))

            current_pdf_status = "No PDF is currently loaded."
            if st.session_state.temp_pdf_retriever:
                current_pdf_status = f"A PDF named '{st.session_state.last_uploaded_filename_processed}' (ID: {st.session_state.temp_pdf_docs[0].metadata.get('source_pdf_id', 'N/A')}) is currently loaded and available for querying."

            agent_output = agent_executor.invoke({
                "input": f"Load this PDF: {temp_pdf_path}",
                "chat_history": chat_history_for_agent,
                "current_pdf_status": current_pdf_status
            })
            st.session_state.last_uploaded_filename_processed = uploaded_file.name
            st.success(f"PDF processed: {agent_output.get('output', 'Error processing PDF.')}")
        except Exception as e:
            st.error(f"Error processing uploaded PDF: {e}")
            traceback.print_exc()
        finally:
            os.unlink(temp_pdf_path)
        st.rerun() 

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask a question or critique an uploaded paper..."):
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        chat_history_for_agent = []
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                chat_history_for_agent.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                chat_history_for_agent.append(AIMessage(content=msg["content"]))

        current_pdf_status = "No PDF is currently loaded."
        if st.session_state.temp_pdf_retriever:
            current_pdf_status = f"A PDF named '{st.session_state.last_uploaded_filename_processed}' (ID: {st.session_state.temp_pdf_docs[0].metadata.get('source_pdf_id', 'N/A')}) is currently loaded and available for querying."

        try:
            agent_output = agent_executor.invoke({
                "input": prompt,
                "chat_history": chat_history_for_agent,
                "current_pdf_status": current_pdf_status
            })
            final_answer = agent_output.get("output", "I could not generate a response for your query.")

            # Simulate streaming by splitting the response
            for chunk in final_answer.split(" "):
                full_response += chunk + " "
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)

        except Exception as e:
            logging.error(f"Error during agent invocation: {e}")
            traceback.print_exc()
            full_response = f"An error occurred while processing your request: {e}. Please try again or rephrase."
            message_placeholder.markdown(full_response)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})
