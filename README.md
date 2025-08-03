# Neuro Critics Agent 🔬
**[Neuro-Critics - Live link](https://neuro-critics-ai.streamlit.app/)**

If the response didn't work (maybe the API key got expired) I am sharing some working demo
<img width="1686" height="874" alt="image" src="https://github.com/AbdulWahab740/Neuro-Critics-AI/blob/main/demo/respond_with_pdf.PNG" />

Other image is in the end!!
---

# **Overview**

The **Neuro Critics AI** is a RAG based Agentic flow app which retrieve Neurological informations from vector store. <br>
The Vector-store comparised of 100+ articles on several topics of neurology. The application designed to assist researchers and enthusiasts in the neurology field in biomedical, particularly focusing on neurological disorders like Parkinson's Disease, Epilepsy, and Alzheimer's Disease and Diagnositics. Leveraging **Retrieval-Augmented Generation (RAG)** and a LangChain agent, this tool empowers users to:

* **Query a pre-built knowledge base** on specified neurological topics.
* **Upload and dynamically analyze PDF research papers** for critique or specific information extraction.
* Engage in a conversational interface that intelligently decides whether to retrieve information from the uploaded PDF or the general knowledge base based on the query context.

This application provides a powerful way to interact with complex scientific literature, offering quick summaries, critical insights, and relevant background information directly within your browser.

---

**What Neuro-topics I scraped?**

**Alzheimer_Disease**, **Stroke_Management**, **Epilepsy**, **Parkinson**, **Diagnostic**, **Neurotransmitter**, **Sclerosis**, **Migraine**, **Neurodevelopmental_disorder**, **TBI**, **Amyotrophic_Lateral_Sclerosis**, **Neuroinflammation**, **Sleep_disorder**, **Brain**, **Social_neurology**  

---

# **Features**

* **Intelligent Agent**: A **LangChain agent** orchestrates interactions, intelligently selecting the most appropriate tool (PDF analysis or knowledge base query) based on user intent and conversational context.
* **Dynamic PDF Processing**: Upload any PDF research paper for on-the-fly analysis. The agent creates a **temporary, in-memory FAISS retriever** specifically for the uploaded document, allowing immediate querying.
* **Contextual Critique**: Ask the agent to **critique or summarize specific aspects** of the currently uploaded PDF.
* **Knowledge Base Integration**: Query a pre-existing **FAISS vector store** containing neuro-related research data for background information or general questions.
* **Smart Context Switching**: The agent is designed to **intelligently differentiate** between queries pertaining to the uploaded PDF and general queries that should be answered from the main knowledge base.
* **Topic Classification**: Automatically classifies user queries to filter knowledge base searches, ensuring more relevant retrievals.
* **Conversational Memory**: Maintains **chat history within the current session**, enabling coherent follow-up questions and a natural conversation flow.
* **User-Friendly Interface**: Built with **Streamlit** for an intuitive and interactive web application experience.

---

# **Technologies Used**

* **Python 3.12**
* **Streamlit**: For building the interactive web frontend.
* **LangChain**: For agent orchestration, prompt management, and tool integration.
* **HuggingFace Embeddings (`all-MiniLM-L6-v2`)**: For generating vector embeddings for text content.
* **FAISS**: For efficient similarity search and vector storage (used for both the persistent knowledge base and the temporary PDF index).
* **PyMuPDFLoader**: For robust loading and parsing of PDF documents.
* **Google Gemini (`gemini-2.0-flash`)**: As the underlying Large Language Model for generating responses and powering agent reasoning.
---

**Setup and Installation** 

Follow these steps to get the Neuro Critics RAG & Critique Agent up and running on your local machine.

**1. Clone the Repository**

First, clone the project repository to your local machine:

```
git clone <your-repo-url>
cd your-project-folder # Replace with your actual project folder name
```
 **2. Create a Virtual Environment (Recommended)**
It's highly recommended to use a virtual environment to manage project dependencies:

```
python -m venv neuro_env
```
Activate the virtual environment:

On macOS/Linux:

```
source neuro_env/bin/activate
```

On Windows:

```
.\neuro_env\Scripts\activate
```

**3. Install Dependencies**
Install all required Python packages using pip:
```
pip install -r requirements.txt
```

Note: If a requirements.txt file is not provided in the repository, you can manually install the core libraries:

```pip install streamlit langchain langchain-community langchain-google-genai pypdfium2-team==2.39.0 pydantic==2.7.1 faiss-cpu python-dotenv numpy
```

# **4. Set Up Environment Variables**
Create a file named .env in the root directory of your project (where streamlit_rag_app.py is located) and add your Google Gemini API key:
```
GEMINI_API_KEY="YOUR_GEMINI_API_KEY_HERE"
```

Important: Replace "YOUR_GEMINI_API_KEY_HERE" with your actual API key obtained from Google AI Studio.

**5. Build the FAISS Knowledge Base**
The application relies on a pre-built FAISS index for its general knowledge base. You need to run the vector_store_builder.py script to create and populate this index. Ensure you have your data sources ready as configured within your data_fetch.py file.
```
python vector_store_builder.py
```
This script will create a faiss_index directory in your project containing the necessary vector store files.

Usage
Once the setup is complete and the FAISS index is built, you can launch the Streamlit application.

1. Run the Streamlit App
Navigate to your project's root directory in the terminal (where streamlit_rag_app.py is located) and execute:

```
streamlit run streamlit_rag_app.py
```

This command will start the Streamlit server and automatically open the application in your default web browser.

2. Interact with the Agent
Chat with the Knowledge Base: Ask general questions about the pre-defined neurological topics (e.g., "What are the common symptoms of Parkinson's?", "Tell me about epilepsy treatments", "What is Alzheimer's disease?"). The agent will query the main FAISS knowledge base for answers.

Upload a PDF: Use the "Choose a PDF file" input and the "⬆️ Upload PDF" button located directly within the main chat input area. The application will process the PDF, and a confirmation message will appear in the chat.

Critique or Query Uploaded PDF: After a PDF is successfully loaded, you can ask specific questions about that paper (e.g., "Critique this paper", "Summarize the methodology", "What are the key findings in this document?"). The agent will intelligently prioritize using the temporary PDF retriever for these queries.

Intelligent Context Switching: The agent is designed to seamlessly switch between querying the uploaded PDF and the main knowledge base. If a PDF is loaded, it will use it for specific paper-related questions. However, for general queries (e.g., "What is the latest research on neurodegeneration?"), it will revert to using the main knowledge base.

Store PDF: You can instruct the agent to "Store this paper" or "Save this PDF" to add the content of the currently loaded temporary PDF to the main persistent FAISS knowledge base.

Project Structure
This project is structured for clarity and maintainability, especially for the Streamlit application:

neuro-critics-rag-agent/
├── streamlit_rag_app.py    # The main Streamlit application file, handling UI, agent setup, and core logic.
├── vector_store_builder.py # Script responsible for building and populating the primary FAISS knowledge base.
├── data_fetch.py           # Contains functions for fetching or defining the structured data sources and topics used in the knowledge base.
├── faiss_index/            # Directory where the persistent FAISS vector store files are stored after building.
│   └── ...                 # (e.g., index.faiss, index.pkl)
├── models/                 # (Optional) Directory for caching HuggingFace embedding model files.
│   └── ...                 # (e.g., model.safetensors, tokenizer.json)
├── .env                    # Environment variables file (e.g., for GEMINI_API_KEY).
└── requirements.txt        # Lists all Python dependencies required for the project.

<img width="1686" height="874" alt="image" src="https://github.com/AbdulWahab740/Neuro-Critics-AI/blob/main/demo/chat_with_knowledge_based.PNG" />

