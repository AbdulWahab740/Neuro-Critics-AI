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
import faiss
import shutil
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logging.info("Attempting to load embedding model, FAISS index, and LLM...")
embedding_model = load_embedding_model()
faiss_index = load_faiss_index(embedding_model)
llm = load_llm()
topic_embeddings = pre_embed_topics(embedding_model)
logging.info("All core components (embedding, FAISS, LLM, topic embeddings) loaded.")

# Ensure required session state keys
if "messages" not in st.session_state:
    st.session_state.messages = []
if "temp_pdf_docs" not in st.session_state:
    st.session_state.temp_pdf_docs = []
if "temp_pdf_retriever" not in st.session_state:
    st.session_state.temp_pdf_retriever = None
if "last_uploaded_filename_processed" not in st.session_state:
    st.session_state.last_uploaded_filename_processed = ""
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False
# Initialize FAISS + metadata only once per session
# if "faiss_index" not in st.session_state:
#     st.session_state.faiss_index = faiss.IndexFlatL2(512)  # dimension
#     st.session_state.metadata_store = []

# if "pdf_processed" not in st.session_state:
#     st.session_state.pdf_processed = False

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
# Constants
MAX_FILE_SIZE_MB = 50
BASE_TMP_DIR = "/tmp/neurocritics"

# Ensure base temp dir exists
os.makedirs(BASE_TMP_DIR, exist_ok=True)

with st.sidebar:
    st.header("Upload PDF for Critique")
    uploaded_file = st.file_uploader(
        "Choose a PDF file", type="pdf", accept_multiple_files=False, key="pdf_uploader"
    )

    if uploaded_file is not None:
        file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
        if file_size_mb > MAX_FILE_SIZE_MB:
            st.error(f"❌ File too large: {file_size_mb:.2f} MB. Limit is {MAX_FILE_SIZE_MB} MB.")
            uploaded_file = None

    if uploaded_file is not None:
        st.write(f"📑 Uploaded: {uploaded_file.name}")

        # Reset state if new file uploaded
        if uploaded_file.name != st.session_state.last_uploaded_filename_processed:
            st.session_state.pdf_processed = False
            st.session_state.last_uploaded_filename_processed = uploaded_file.name
            st.session_state.temp_pdf_docs = []
            st.session_state.temp_pdf_retriever = None

            # 🚮 Clear old files to save space
            if os.path.exists(BASE_TMP_DIR):
                shutil.rmtree(BASE_TMP_DIR)
            os.makedirs(BASE_TMP_DIR, exist_ok=True)

        # Process only if not already processed
        if not st.session_state.pdf_processed:
            st.info(f"📄 Processing: {uploaded_file.name}")

            # Save uploaded file to /tmp/neurocritics
            temp_pdf_path = os.path.join(BASE_TMP_DIR, uploaded_file.name)
            with open(temp_pdf_path, "wb") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())

            # Extract images directory
            output_dir = os.path.join(BASE_TMP_DIR, uploaded_file.name[:10] + "_extracted_images")
            os.makedirs(output_dir, exist_ok=True)

            try:
                with st.spinner("🔍 Extracting images from PDF..."):
                    extract_images(temp_pdf_path, output_dir)

                with st.spinner("⚡ Processing images and building index..."):
                    process_images_and_build_index(output_dir)

                # 🔑 Save docs/retriever in session state if needed
                # st.session_state.temp_pdf_retriever = retriever
                # st.session_state.temp_pdf_docs = docs  

                st.session_state.pdf_processed = True
                st.success("✅ PDF successfully processed!")

            except Exception as e:
                st.error(f"Error processing uploaded PDF: {e}")
                traceback.print_exc()

        else:
            st.info("This PDF is already processed and ready for querying.")
# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

text_query = st.chat_input("Ask a question or critique an uploaded paper...:")

if text_query:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": text_query})

    with st.chat_message("user"):
        st.markdown(text_query)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()

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
                "input": text_query,
                "chat_history": chat_history_for_agent,
                "current_pdf_status": current_pdf_status
            })
            final_answer = agent_output.get("output", "I could not generate a response for your query.")

            # 👇 Use markdown so headings, bullets, etc. show correctly
            message_placeholder.markdown(final_answer, unsafe_allow_html=True)

            # Also save in chat history for continuity
            st.session_state.messages.append({"role": "assistant", "content": final_answer})

        except Exception as e:
            st.error(f"Error: {e}")


        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": final_answer})








