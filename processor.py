import os
import requests
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

# -------------------------
# GLOBAL VARIABLES
# -------------------------

vector_db = None
full_text = ""

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)


# -------------------------
# LLM CALL
# -------------------------

def call_llm(system, user):

    api_key = os.getenv("OPENROUTER_API_KEY")

    url = "https://openrouter.ai/api/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "nvidia/nemotron-3-super-120b-a12b-20230311:free",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ]
    }

    try:

        res = requests.post(url, headers=headers, json=data, timeout=60, verify=False)

        response_json = res.json()

        response_data = response_json.get("choices")

        if response_data:
            return response_data[0]["message"]["content"], "nvidia/nemotron-3-super-120b"

        error_message = response_json.get("error", {}).get("message", "AI Busy")

        return f"Limit Reached: {error_message}", "Error"

    except Exception as e:

        return f"Connection Failed: {str(e)}", "Network Error"


# -------------------------
# DOCUMENT INGESTION
# -------------------------

def ingest_document(file_path):

    global vector_db, full_text

    loader = PyPDFLoader(file_path)

    pages = loader.load()

    full_text = " ".join([p.page_content for p in pages])

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(pages)

    vector_db = FAISS.from_documents(chunks, embeddings)

    vector_db.save_local("faiss_index")

    return "Knowledge Base ready in RAM and saved to disk!"


# -------------------------
# TOOL 1 — SUMMARIZER
# -------------------------

def summarize_tool(query):

    system_msg = (
        "You are a technical document analyst. Summarize the document using the following format:\n"
        "1. Use a clear Heading for the summary.\n"
        "2. Use a bulleted list for key functionalities.\n"
        "3. **Bold** all function or API names.\n"
        "4. End with a one-sentence 'Core Objective' of the document."
    )

    ans, model_name = call_llm(system_msg, f"Summarize this: {full_text[:12000]}")

    return ans, "Summarize Tool", "Full Document", model_name


# -------------------------
# TOOL 2 — VISUALIZATION
# -------------------------

def visualization_tool(query):

    global vector_db

    docs = vector_db.similarity_search(query, k=3)

    context = "\n".join([d.page_content for d in docs])

    system_msg = (
        "You are a technical architect. Create a visualization of the process described in the context.\n"
        "1. Use a Mermaid.js 'graph TD' flowchart to show the logic.\n"
        "2. If the data is better suited for a table, provide a Markdown table.\n"
        "3. Provide a brief 2-sentence explanation of the visualization.\n"
        "4. Wrap Mermaid code in ```mermaid blocks."
    )

    ans, model_name = call_llm(
        system_msg,
        f"Context: {context}\nQuery: Create a visualization for: {query}"
    )

    citations = f"Pages: {', '.join(sorted(list(set([str(d.metadata.get('page',0)+1) for d in docs]))))}"

    return ans, "Visualization Tool", citations, model_name


# -------------------------
# TOOL 3 — GENERAL Q&A
# -------------------------

def qa_tool(query):

    global vector_db

    docs = vector_db.similarity_search(query, k=3)

    context = "\n".join([d.page_content for d in docs])

    page_numbers = sorted(list(set([str(d.metadata.get('page',0)+1) for d in docs])))

    citations = f"Pages: {', '.join(page_numbers)}"

    system_msg = (
        "You are a helpful technical assistant. Answer the user's question using the provided context.\n"
        "Format your response as follows:\n"
        "- Start with a direct answer in 1-2 sentences.\n"
        "- Use a bulleted list if explaining multiple steps or functions.\n"
        "- Use **bold** for API names and `code blocks` for parameters.\n"
        "- If the answer isn't in the context, clearly state 'Information not found in document.'"
    )

    ans, model_name = call_llm(system_msg, f"Context: {context}\nQuery: {query}")

    return ans, "General Q&A Tool", citations, model_name


# -------------------------
# AGENT DISPATCHER
# -------------------------

def agent_dispatcher(query):

    query_lower = query.lower()

    if "summarize" in query_lower:
        return summarize_tool(query)

    elif any(word in query_lower for word in ["visualize", "flowchart", "diagram", "chart", "map"]):
        return visualization_tool(query)

    else:
        return qa_tool(query)
