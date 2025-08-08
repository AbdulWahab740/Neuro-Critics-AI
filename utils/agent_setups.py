import logging
import traceback
import streamlit as st
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate
from utils.llm_loader import load_llm
from tools.pdf_tools import load_pdf_and_create_temp_retriever, query_temp_pdf_for_critique
from tools.query_tools import query_main_knowledge_base
from tools.image_caption_tool import search_caption_with_query
llm = load_llm()

@st.cache_resource
def setup_agent():
    """Sets up the agent with the necessary tools and prompt."""
    tools = [
        load_pdf_and_create_temp_retriever,
        query_temp_pdf_for_critique,
        query_main_knowledge_base,
        search_caption_with_query
    ]
    logging.info(f"Defined {len(tools)} tools for the agent.")

    # Modified agent_prompt to include MessagesPlaceholder for chat history
    agent_prompt = PromptTemplate.from_template(
        """You are an intelligent AI assistant capable of analyzing biomedical research,
        loading new PDF documents, querying a knowledge base, and performing critical reviews.
        
        You have access to the following tools:

        - `load_pdf_and_create_temp_retriever`: Load and index a PDF into temporary memory.
        - `query_temp_pdf_for_critique`: Query and critique the content of the currently loaded PDF.
        - `query_main_knowledge_base`: Retrieve answers from the broader knowledge base on neurological topics.
        - `search_caption_with_query`: Search for diagrams, images, or figures based on a natural language question by matching it to image captions extracted from the PDF.

        Tool usage guide:

        - If the 'Current PDF Status' indicates a PDF is already loaded:
            - Use `query_temp_pdf_for_critique` for **textual critiques, summaries, or methods** in the paper.
            - Use `search_caption_with_query` if the user refers to a **diagram, image, or figure** (e.g., "show me the diagram about tau protein", "what does the image say on page 3?").
            - Use `query_main_knowledge_base` if the user asks a general question **not specific to the loaded PDF** (e.g., "What causes migraines?", "Explain ALS symptoms").
        - DO NOT ask for a PDF path again if one is already loaded.

        Filter topics to match user query with knowledge base: 
        ['Alzheimer_Disease', 'Stroke_Management', 'Epilepsy', 'Parkinson', 'Diagnostic', 'Neurotransmitter', 'Sclerosis', 'Migraine', 'Neurodevelopmental_disorder', 'TBI', 'Amyotrophic_Lateral_Sclerosis', 'Neuroinflammation', 'Sleep_disorder', 'Brain', 'Social_neurology']

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