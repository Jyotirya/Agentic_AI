# 🚀 Quick Start Guide

Welcome to the IITB EE Department RAG Assistant!

## Prerequisites

- Python 3.8+
- Google API Key (get from https://makersuite.google.com/app/apikey)

## Installation (5 minutes)

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up environment:**
   ```bash
   cp .env.example .env
   # Edit .env and add your GOOGLE_API_KEY
   ```

3. **Initialize the system:**
   ```bash
   python main.py --init
   ```

## Running the Application

### Web Interface (Recommended)
```bash
streamlit run app.py
# or
python main.py --web
```

### CLI Interactive Mode
```bash
python main.py
```

### Single Query
```bash
python main.py --query "What are the prerequisites for EE720?"
```

## Example Queries

Try these questions:
- "What are the prerequisites for EE720?"
- "Who teaches Advanced Power Electronics?"
- "What are Prof. Amit Verma's research interests?"
- "Are there any upcoming workshops?"
- "Which faculty work on wireless communication?"

## Features Overview

✅ **Smart Search** - Semantic search with reranking
✅ **Rich UI** - Chat interface with sources and feedback
✅ **Persistent Storage** - Fast startup with ChromaDB
✅ **Multi-agent** - Specialized agents for each task
✅ **Configurable** - Easy customization via .env

## Troubleshooting

**Problem:** "Google API key not found"
**Solution:** Create `.env` file with your API key

**Problem:** "No documents found"
**Solution:** Ensure JSON files exist in `data/raw/`

**Problem:** Memory error
**Solution:** Reduce batch size in config

## Need Help?

- Check the full README.md for detailed documentation
- Run `python main.py --config` to see current settings
- Run tests with `python tests/test_rag.py`

## What's New in This Version?

🎉 **Enhanced Features:**
- Advanced chunking strategies (4 types)
- Cross-encoder reranking for better accuracy
- Persistent ChromaDB storage
- Comprehensive prompt engineering
- Rich Streamlit UI with feedback
- Full configuration system
- Test suite included

---

**Ready to explore? Start with `streamlit run app.py`**
