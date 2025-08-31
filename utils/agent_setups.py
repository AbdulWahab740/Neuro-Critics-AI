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
        query_temp_pdf_for_critique,
        query_main_knowledge_base,
        search_caption_with_query
    ]
    logging.info(f"Defined {len(tools)} tools for the agent.")

    # Refined prompt with graceful stop condition
    agent_prompt = PromptTemplate.from_template(
    """You are an intelligent biomedical research assistant.
You can analyze research, load new PDFs, query knowledge bases, and critique content.

**Available Tools:**
- `query_temp_pdf_for_critique`: Query and critique the content of the currently loaded PDF.
- `query_main_knowledge_base`: Retrieve answers from the broader knowledge base on neurological topics.
- `search_caption_with_query`: Search for diagrams, images, or figures by matching them to image captions.

**How to decide which tool to use:**
- If a PDF is already loaded:
  - Use `query_temp_pdf_for_critique` for text, summaries, methods, or critiques.
  - Use `search_caption_with_query` if the user refers to a diagram/figure. Use the provided page number as query! - Return the well-structured biomedical explanation for the user
  - Use `query_main_knowledge_base` for general medical/neurological questions not tied to the PDF.
- If no PDF is loaded and the user asks a general question, use `query_main_knowledge_base`.

**Important Behavioral Rules:**
- When you receive an answer from any tool, return it *exactly as given but not for `search_caption_with_query`*, without shortening or summarizing.
- Preserve all formatting (headings, bullet points, paragraphs).
- Do not paraphrase, do not compress — just deliver the tool’s output as-is.
- If the tool output is empty, then politely say you couldn’t find an answer.
- If you use `search_caption_with_query`, take its output (caption + metadata) and expand it into a structured biomedical explanation for the user.

Current PDF Status: {current_pdf_status}

Previous conversation:
{chat_history}

User Question:
{input}

{agent_scratchpad}
"""
)

    try:
        agent = create_tool_calling_agent(llm, tools, agent_prompt)
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True
        )
        logging.info("Agent Executor initialized successfully.")
        return agent_executor
    except Exception as e:
        logging.error(f"Failed to setup agent: {e}")
        traceback.print_exc()
        st.error(f"Failed to setup agent: {e}. Check LLM and tool definitions.")
        st.stop()



