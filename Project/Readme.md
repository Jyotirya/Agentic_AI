# 📚 IITB Electrical Engineering Department RAG Assistant

A sophisticated Retrieval-Augmented Generation (RAG) system built with LangGraph for answering queries about the IITB Electrical Engineering Department. This intelligent assistant provides information about courses, faculty, research activities, and departmental announcements.

## 🌟 Features

- **Multi-Agent Architecture**: Modular design with specialized agents for scraping, embedding, querying, and response generation
- **Advanced Retrieval**: Semantic search with cross-encoder reranking for improved accuracy
- **Persistent Storage**: ChromaDB vector database with persistent storage
- **Multiple Chunking Strategies**: Recursive, semantic, fixed-size, and adaptive chunking options
- **Rich UI**: Interactive Streamlit interface with conversation history, source citations, and feedback system
- **Query Classification**: Automatic detection of query types (course, faculty, research, announcement)
- **Conditional Workflow**: LangGraph-based orchestration with error handling and recovery
- **Comprehensive Prompts**: Specialized prompts for different query types

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                       Streamlit UI                          │
│        (Conversation History, Sources, Feedback)            │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                  LangGraph Supervisor                       │
│  (Conditional Routing, State Management, Error Handling)    │
└─────┬─────────┬─────────┬─────────┬─────────────────────────┘
      │         │         │         │
      ▼         ▼         ▼         ▼
┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────────┐
│ Scraper │ │Embedding│ │  Query  │ │  Response   │
│  Agent  │ │  Agent  │ │  Agent  │ │   Agent     │
└─────────┘ └─────────┘ └─────────┘ └─────────────┘
     │           │           │             │
     ▼           ▼           ▼             ▼
┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────────┐
│  JSON   │ │ChromaDB │ │ Reranker│ │Google Gemini│
│  Data   │ │ Vector  │ │ (Cross- │ │     API     │
│         │ │   DB    │ │Encoder) │ │             │
└─────────┘ └─────────┘ └─────────┘ └─────────────┘
```

## 📦 Project Structure

```
Project/
├── app.py                      # Streamlit application
├── requirements.txt            # Python dependencies
├── Readme.md                   # This file
│
├── agents/                     # Agent modules
│   ├── scraper_agent.py       # Data loading and preprocessing
│   ├── embedding_agent.py     # Embedding generation and storage
│   ├── query_agent.py         # Semantic search and reranking
│   └── response_agent.py      # LLM response generation
│
├── graph/                      # Workflow orchestration
│   └── supervisor.py          # LangGraph workflow definition
│
├── utils/                      # Utility modules
│   ├── chunking.py            # Document chunking strategies
│   └── prompts.py             # Prompt templates
│
└── data/                       # Data directory
    ├── raw/                   # Source data
    │   ├── courses.json       # Course information
    │   ├── faculty.json       # Faculty profiles
    │   └── announcements.json # Department announcements
    └── chroma_db/             # Persistent vector database (auto-created)
```

## 🚀 Setup Instructions

### Prerequisites

- Python 3.8 or higher
- Google API Key (for Gemini LLM)

### Installation

1. **Clone or navigate to the project directory:**
   ```bash
   cd "Agentic Ai/Project"
   ```

2. **Create and activate a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   Create a `.env` file in the Project directory:
   ```bash
   GOOGLE_API_KEY=your_google_api_key_here
   ```

   To get a Google API key:
   - Visit https://makersuite.google.com/app/apikey
   - Create or select a project
   - Generate an API key

5. **Verify data files:**
   Ensure the `data/raw/` directory contains:
   - `courses.json`
   - `faculty.json`
   - `announcements.json`

## 💻 Usage

### Running the Application

Start the Streamlit app:
```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

### Using the Interface

1. **Ask Questions**: Type your question in the chat input at the bottom
2. **View Answers**: See contextual answers with confidence scores
3. **Check Sources**: Click "View Sources" to see retrieved context chunks
4. **Provide Feedback**: Use 👍/👎 buttons to rate responses
5. **Export History**: Download conversation history as JSON
6. **Clear History**: Reset the conversation anytime

### Example Queries

**Course Information:**
- "What are the prerequisites for EE720?"
- "Which courses cover machine learning?"
- "Tell me about the Digital Systems course"

**Faculty Information:**
- "Who teaches Advanced Power Electronics?"
- "Which faculty work on wireless communication?"
- "What are Prof. Amit Verma's research interests?"

**Research Queries:**
- "What research is happening in power electronics?"
- "Who should I contact for a PhD in VLSI design?"

**Announcements:**
- "Are there any upcoming workshops?"
- "When is the registration deadline?"
- "Tell me about Vidyut 2026"

## 🔧 Configuration

### Chunking Strategy

Modify in `graph/supervisor.py`:
```python
chunks, chunk_metadata = chunk_documents_with_metadata(
    docs=docs,
    chunk_size=500,        # Adjust chunk size
    chunk_overlap=100,     # Adjust overlap
    strategy="recursive"   # Options: recursive, semantic, fixed, adaptive
)
```

### Retrieval Parameters

Modify in `graph/supervisor.py`:
```python
context, metadata, scores = retrieve_context_with_metadata(
    collection=collection,
    query=question,
    k=5                    # Number of chunks to retrieve
)
```

### Model Selection

Change in `agents/response_agent.py`:
```python
self.model = genai.GenerativeModel("gemini-1.5-flash")  # or gemini-pro
```

## 🛠️ Advanced Features

### Custom Data Sources

Add new JSON files to `data/raw/`:
```json
{
  "field1": "value1",
  "field2": "value2"
}
```

The scraper automatically processes all JSON files in the directory.

### Extending Agents

Each agent is modular and can be extended:

**Scraper Agent**: Add web scraping capabilities
```python
from agents.scraper_agent import DocumentScraper
scraper = DocumentScraper()
scraper.scrape_from_web(url)
```

**Embedding Agent**: Use different embedding models
```python
from agents.embedding_agent import EmbeddingManager
manager = EmbeddingManager(model_name="all-mpnet-base-v2")
```

**Query Agent**: Enable hybrid search
```python
from agents.query_agent import QueryManager
manager = QueryManager()
docs, meta, scores = manager.hybrid_search(collection, query, k=5)
```

### Custom Prompts

Add specialized prompts in `utils/prompts.py`:
```python
CUSTOM_PROMPT = """
Your custom system prompt here...
"""
```

## 📊 Performance Optimization

### Vector Database

- **Persistent Storage**: ChromaDB stores embeddings on disk for fast startup
- **Batch Processing**: Documents are embedded and stored in batches
- **Incremental Updates**: New documents can be added without rebuilding

### Embedding

- **Model**: `all-MiniLM-L6-v2` (fast, efficient)
- **Batch Size**: 32 (adjustable)
- **Dimension**: 384

### Reranking

- **Cross-Encoder**: `ms-marco-MiniLM-L-6-v2`
- **Improves retrieval accuracy by 15-20%**
- **Can be disabled for faster responses**

## 🐛 Troubleshooting

### Common Issues

**Issue**: "ChromaDB collection not found"
- **Solution**: Delete `data/chroma_db/` and restart the app to rebuild

**Issue**: "Google API key not found"
- **Solution**: Create `.env` file with `GOOGLE_API_KEY=your_key`

**Issue**: "Module not found"
- **Solution**: Ensure all dependencies are installed: `pip install -r requirements.txt`

**Issue**: "Memory error during embedding"
- **Solution**: Reduce batch size in `embedding_agent.py`

### Logging

Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📚 Dependencies

Key libraries:
- **streamlit**: Web UI framework
- **langchain**: Text processing and splitting
- **langgraph**: Workflow orchestration
- **chromadb**: Vector database
- **sentence-transformers**: Embedding models
- **google-generativeai**: Gemini LLM
- **python-dotenv**: Environment management
- **tqdm**: Progress bars

## 🤝 Contributing

To extend or improve the project:

1. Add new agents in `agents/` directory
2. Extend the workflow in `graph/supervisor.py`
3. Add new prompt templates in `utils/prompts.py`
4. Enhance the UI in `app.py`
5. Add new data sources in `data/raw/`

## 📝 License

This project is for educational purposes as part of the Agentic AI coursework.

## 👥 Support

For issues or questions:
- Check the troubleshooting section
- Review code comments and docstrings
- Refer to library documentation

## 🎯 Future Enhancements

Potential improvements:
- [ ] Add authentication and user management
- [ ] Implement conversation memory across sessions
- [ ] Add multimodal support (images, PDFs)
- [ ] Deploy to cloud platform
- [ ] Add real-time data updates
- [ ] Implement A/B testing for prompts
- [ ] Add analytics dashboard
- [ ] Support multiple languages

---

**Built with ❤️ using LangGraph, ChromaDB, and Google Gemini**
