# 🤖 Agentic RAG Assistant

A premium AI-powered document analysis and question-answering system built with Flask, LangChain, and FAISS. Upload PDF documents and ask intelligent questions with context-aware responses powered by advanced language models.


https://github.com/user-attachments/assets/6207e949-8ce8-4c31-bc7a-b243f0b7b520




## ✨ Features

- **📄 Document Upload** - Drag-and-drop PDF file upload
- **🔍 Intelligent Retrieval** - FAISS vector database for semantic search
- **🤖 AI-Powered Q&A** - Context-aware answers using language models
- **💬 Chat Interface** - ChatGPT-like conversation experience
- **🎨 Dark/Light Theme** - Beautiful UI with theme toggle
- **📌 Source Citations** - Automatic citation of relevant document pages
- **⚡ Real-time Processing** - Instant document embedding and indexing
- **📱 Responsive Design** - Works seamlessly on desktop and mobile

## 🛠️ Tech Stack

- **Backend**: Python, Flask
- **AI/ML**: LangChain, FAISS, Google Gemini API Embeddings
- **LLM**: OpenRouter (Free tier with multiple models)
- **Frontend**: HTML5, CSS3, JavaScript (Vanilla)
- **Vector Database**: FAISS (Facebook AI Similarity Search)
- **Document Processing**: PyPDF, LangChain Document Loaders

## 📋 Prerequisites

- Python 3.8+
- pip (Python package manager)
- A valid OpenRouter API key (free tier available)

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/agentic-rag-assistant.git
cd agentic-rag-assistant
```

### 2. Create Virtual Environment
```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables
Create a `.env` file in the project root:
```
OPENROUTER_API_KEY=your_api_key_here
```

Get your API key from [OpenRouter](https://openrouter.ai/)

## 📖 Usage

### 1. Start the Application
```bash
python app.py
```

The app will be available at `http://127.0.0.1:5000`

### 2. Upload a Document
- Click "📁 Click or drag and drop" in the Documents panel
- Select a PDF file or drag it into the upload area
- Click "Process Document"
- Wait for the knowledge base to be indexed

### 3. Ask Questions
- Once the document is uploaded, use the "Ask a Question" section
- Type your question about the document
- Press Enter or click "Send"
- Get instant AI-powered responses with citations

## 📁 Project Structure

```
agentic-rag-assistant/
├── app.py                 # Flask application & API routes
├── requirements.txt       # Python dependencies
├── .gitignore            # Git ignore rules
├── .env                  # Environment variables (not committed)
├── README.md             # This file
│
├── templates/
│   └── index.html        # Frontend interface
│
├── static/
│   └── style.css         # Styling & theming
│
├── uploads/              # User uploaded PDFs (generated)
│   └── *.pdf
│
└── venv/                 # Virtual environment (not committed)
```

## 🔧 Configuration

### Environment Variables
- `OPENROUTER_API_KEY` - Your OpenRouter API key for LLM access

### Key Settings (in app.py)
- `UPLOAD_FOLDER` - Directory for uploaded PDFs (default: `uploads/`)
- `Embedding Model` - Google Gemini API: `gemini-embedding-001`
- `Chunk Size` - Document chunk size: `1000` tokens
- `Chunk Overlap` - Overlap between chunks: `100` tokens
- `Search Results` - Number of retrieved chunks: `3`

## 🎯 How It Works

1. **Document Processing**
   - PDF is loaded and split into chunks
   - Chunks are embedded using Google Gemini API embeddings
   - Embeddings are stored in FAISS vector database

2. **Query Processing**
   - User query is embedded using the same model
   - Semantic similarity search retrieves relevant chunks
   - Retrieved context is passed to the LLM

3. **Response Generation**
   - LLM generates answer based on context
   - Response includes tool used and source citations
   - Answer displayed in real-time chat interface

## 🌐 Browser Support

- Chrome/Edge 90+
- Firefox 88+
- Safari 14+
- Mobile browsers (iOS Safari, Chrome Mobile)

## ⚙️ API Endpoints

### `POST /upload`
Upload and process a PDF document
- **Body**: Form data with `file` field (PDF)
- **Response**: Success status and message

### `POST /ask`
Submit a question about the uploaded document
- **Body**: JSON with `query` field
- **Response**: JSON with `answer`, `tool`, `model`, `citations`

### `GET /`
Serve the main HTML interface

## 🚨 Troubleshooting

### PDF Upload Errors
- Ensure file is a valid PDF
- Check file size (recommended: <50MB)
- Verify permissions in uploads directory

### API Rate Limiting
- OpenRouter free tier has rate limits
- Consider upgrading for production use
- Check API status at [OpenRouter](https://openrouter.ai/)

## 📝 Example Usage

1. **Upload a technical document** (e.g., API documentation)
2. **Ask specific questions** like:
   - "Summarize the key features"
   - "What are the main classes and their methods?"
   - "Explain how the authentication works"
3. **Get instant answers** with source page references

## 🔐 Security Notes

- API keys are stored in `.env` (never commit)
- `.gitignore` prevents accidental key exposure
- Uploaded files are stored locally
- No data is retained after session ends
- Use environment variables for sensitive data

## 📈 Performance Tips

- Use PDFs with clear text (avoid scanned images)
- Optimal document size: 1-50 pages
- Larger documents take longer to process
- Adjust `chunk_size` for better context preservation

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- [LangChain](https://www.langchain.com/) - LLM framework
- [FAISS](https://github.com/facebookresearch/faiss) - Vector search
- [OpenRouter](https://openrouter.ai/) - LLM API
- [Google Gemini API](https://aistudio.google.com/) - Embedding models
- [Flask](https://flask.palletsprojects.com/) - Web framework


## 🔄 Roadmap

- [ ] Support for more document formats (DOCX, TXT, etc.)
- [ ] Conversation memory and history
- [ ] Document management dashboard
- [ ] Advanced search filters
- [ ] Export conversation transcripts
- [ ] Multi-language support
- [ ] API authentication & rate limiting
- [ ] Docker containerization

---

**Last Updated: March 2026**
