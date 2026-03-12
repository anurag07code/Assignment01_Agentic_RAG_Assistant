from flask import Flask, render_template, request, jsonify
import os
import requests
import json
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FakeEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings 

# Load variables from .env
load_dotenv()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

#embeddings = FakeEmbeddings(size=384)
embeddings = GoogleGenerativeAIEmbeddings(
    # CHANGE THIS LINE:
    model="models/gemini-embedding-001", 
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

vector_db = None
full_text = ""

# --- UPDATED TO RETURN BOTH ANSWER AND MODEL NAME ---
def call_llm(system, user):
    api_key = os.getenv("OPENROUTER_API_KEY")
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    
    data = {
        "model": "nvidia/nemotron-3-super-120b-a12b-20230311:free",
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]
    }
    
    try:
        res = requests.post(url, headers=headers, json=data, timeout=60, verify=False)
        response_json = res.json()

        # 1. Use .get() so it returns None instead of crashing if 'choices' is missing
        response_data = response_json.get('choices')

        # 2. Check if we actually got an answer
        if response_data:
            return response_data[0]['message']['content'], "nvidia/nemotron-3-super-120b"
        
        # 3. If we didn't get an answer, find out why from the error key
        error_message = response_json.get('error', {}).get('message', 'AI Busy')
        return f"Limit Reached: {error_message}", "Error"

    except Exception as e:
        return f"Connection Failed: {str(e)}", "Network Error"
'''def call_llm(system, user):
    api_key = os.getenv("OPENROUTER_API_KEY")
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    
    data = {
        "model": "openrouter/free",
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]
    }
    
    try:
        # Added verify=False to bypass the SSL certificate error
        res = requests.post(url, headers=headers, json=data, timeout=60, verify=False)
        
        # This will hide the "InsecureRequestWarning" in your terminal
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        
        response_json = res.json()
        model_used = response_json.get('model', 'Unknown Free Model')
        content = response_json['choices'][0]['message']['content']
        
        return content, model_used
    except Exception as e:
        return f"LLM Error: {str(e)}", "Error"'''

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global vector_db, full_text
    file = request.files['file']
    if not os.path.exists(app.config['UPLOAD_FOLDER']): os.makedirs(app.config['UPLOAD_FOLDER'])
    
    path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(path)
    
    loader = PyPDFLoader(path)
    pages = loader.load()
    full_text = " ".join([p.page_content for p in pages])
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(pages)
    
    # --- FIX START ---
    # 1. First, create the database in RAM
    vector_db = FAISS.from_documents(chunks, embeddings)
    
    # 2. Then, save it to the disk folder
    vector_db.save_local("faiss_index") 
    # --- FIX END ---
    
    return jsonify({"status": "Success", "message": "Knowledge Base ready in RAM and saved to disk!"})

@app.route('/ask', methods=['POST'])
def ask():
    query = request.json.get('query', '')
    citations = "" 
    
    if "summarize" in query.lower():
        system_msg = (
            "You are a technical document analyst. Summarize the document using the following format:\n"
            "1. Use a clear Heading for the summary.\n"
            "2. Use a bulleted list for key functionalities.\n"
            "3. **Bold** all function or API names.\n"
            "4. End with a one-sentence 'Core Objective' of the document."
        )
        ans, model_name = call_llm(system_msg, f"Summarize this: {full_text[:12000]}")
        tool = "Summarize Tool"
        citations = "Full Document"

    elif any(word in query.lower() for word in ["visualize", "flowchart", "diagram", "chart", "map"]):
        docs = vector_db.similarity_search(query, k=3)
        context = "\n".join([d.page_content for d in docs])
        
        system_msg = (
            "You are a technical architect. Create a visualization of the process described in the context.\n"
            "1. Use a Mermaid.js 'graph TD' flowchart to show the logic.\n"
            "2. If the data is better suited for a table, provide a Markdown table.\n"
            "3. Provide a brief 2-sentence explanation of the visualization.\n"
            "4. Wrap Mermaid code in ```mermaid blocks."
        )
        
        ans, model_name = call_llm(system_msg, f"Context: {context}\nQuery: Create a visualization for: {query}")
        tool = "Visualization Tool"
        citations = f"Pages: {', '.join(sorted(list(set([str(d.metadata.get('page', 0) + 1) for d in docs]))))}"
    else:
        docs = vector_db.similarity_search(query, k=3)
        page_numbers = sorted(list(set([str(d.metadata.get('page', 0) + 1) for d in docs])))
        citations = f"Pages: {', '.join(page_numbers)}"
        
        context = "\n".join([d.page_content for d in docs])
        system_msg = (
            "You are a helpful technical assistant. Answer the user's question using the provided context.\n"
            "Format your response as follows:\n"
            "- Start with a direct answer in 1-2 sentences.\n"
            "- Use a bulleted list if explaining multiple steps or functions.\n"
            "- Use **bold** for API names and `code blocks` for parameters.\n"
            "- If the answer isn't in the context, clearly state 'Information not found in document.'"
        )
        
        ans, model_name = call_llm(system_msg, f"Context: {context}\nQuery: {query}")
        tool = "General Q&A Tool"
    
    # Return the 'model_name' so you can see it in the UI
    return jsonify({
        "answer": ans, 
        "tool": tool, 
        "citations": citations, 
        "model": model_name
    })

if __name__ == '__main__':
    app.run(debug=True)