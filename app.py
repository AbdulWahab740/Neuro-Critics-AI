import streamlit as st
import os
from langchain_core.messages import AIMessage, HumanMessage
import logging
import tempfile
import traceback
from utils.embeddings import load_embedding_model, pre_embed_topics
from utils.faiss_utils import load_faiss_index
from utils.llm_loader import load_llm
from utils.agent_setups import setup_agent
from utils.caption_index_builder import process_images_and_build_index
from utils.pdf_load import extract_images
from voice_chat import record_and_transcribe
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logging.info("Attempting to load embedding model, FAISS index, and LLM...")
embedding_model = load_embedding_model()
faiss_index = load_faiss_index(embedding_model)
llm = load_llm()
topic_embeddings = pre_embed_topics(embedding_model)
logging.info("All core components (embedding, FAISS, LLM, topic embeddings) loaded.")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "temp_pdf_docs" not in st.session_state:
    st.session_state.temp_pdf_docs = []
if "temp_pdf_retriever" not in st.session_state:
    st.session_state.temp_pdf_retriever = None
if "last_uploaded_filename_processed" not in st.session_state:
    st.session_state.last_uploaded_filename_processed = ""

# Initialize the agent executor
agent_executor = setup_agent()
logging.info("Agent setup complete. Starting Streamlit UI rendering...")

# --- Streamlit UI ---
st.set_page_config(page_title="Neuro Critics AI", page_icon="🔬")

st.title("🔬 Neuro Critics AI")
st.markdown(
    """
    A RAG based Agentic flow app which retrieve Neurological informations from vector store. <br>
    The Vector-store comparised of 100+ articles on several neurology-based topics.<br>
    Ask questions about Parkinson's Disease, Epilepsy, Sclerosis, Migraine, ALS, TBI or Alzheimer's Disease..<br>
    It also contian Diagnositics, Brain, Stroke, sleep disorder and Neurotransmitter related infomrations.<br>
    **OR** upload a PDF paper and ask me to 'Critique this paper' or ask specific questions about it. <br>
    You can also ask related to the **images** in the PDF by telling the page number.<br>
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
        
        output =  uploaded_file.name[:10] + "_extracted_images"
        extract_images(temp_pdf_path,output)
        process_images_and_build_index(output)
        
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
            st.success(f"PDF processed Successfully!!")
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

tab1, tab2 = st.tabs(["📄 Chat Input", "🎤 Voice Input"])

prompt = None  # Default prompt
proceed = False  # Flag to control submission

with tab1:
    text_query = st.chat_input("Ask a question or critique an uploaded paper...:")
    if text_query:
        prompt = text_query
        proceed = True

with tab2:
    voice_query = record_and_transcribe()
    if voice_query:
        st.markdown("✏️ **Edit or Confirm your query:**")
        user_query = st.text_area("Your query", value=voice_query, height=50)
        if st.button("Submit Query"):
            prompt = user_query.strip()
            proceed = True

if proceed and prompt:
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
