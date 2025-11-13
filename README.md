# HR Policy RAG System

Production-grade Retrieval-Augmented Generation (RAG) system for HR policy Q&A with conversation memory.

## Features

- **Hybrid Search**: Pinecone native hybrid search (semantic + keyword)
- **Conversation Memory**: Simple buffer memory (last 5 Q&A pairs per session)
- **Fast LLM**: Gemini 2.0 Flash for cost-effective, fast responses
- **Session Management**: Multiple concurrent sessions with automatic timeout
- **Modern UI**: Clean, responsive chat interface
- **Production Ready**: Comprehensive error handling, health checks, monitoring

## Architecture

```
┌─────────────────┐
│   Frontend      │  HTML/CSS/JS chat UI
│  (Static Web)   │
└────────┬────────┘
         │
         ↓ HTTP/REST
┌─────────────────┐
│   FastAPI       │  Session management, CORS
│   Backend       │
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│  RAG Pipeline   │  Orchestrator
└────────┬────────┘
         │
    ┌────┴────┐
    ↓         ↓
┌────────┐ ┌────────┐
│Retriever│ │Generator│
│(Hybrid) │ │(Gemini) │
└────────┘ └────────┘
    │         │
    ↓         ↓
┌────────┐ ┌────────┐
│Pinecone│ │ Memory │
│(Vector)│ │(Buffer)│
└────────┘ └────────┘
```

## Tech Stack

- **Vector DB**: Pinecone (serverless, hybrid search)
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **Sparse Embeddings**: BM25 via pinecone-text
- **LLM**: Google Gemini 2.0 Flash
- **Backend**: FastAPI + Uvicorn
- **Frontend**: Vanilla HTML/CSS/JavaScript
- **Chunking**: LangChain RecursiveCharacterTextSplitter

## Project Structure

```
basic_rag/
├── config/
│   ├── __init__.py
│   └── settings.py          # Centralized configuration
├── src/
│   ├── core/
│   │   ├── __init__.py
│   │   ├── retriever.py     # Hybrid retriever (Pinecone)
│   │   ├── generator.py     # LLM generator (Gemini) + Memory
│   │   └── rag.py           # RAG orchestrator
│   └── api/
│       ├── __init__.py
│       └── main.py          # FastAPI application
├── frontend/
│   ├── index.html           # Chat UI
│   ├── style.css            # Styles
│   └── app.js               # Frontend logic
├── data/
│   └── chunks.json          # Processed document chunks
├── .env                     # API keys (not in git)
├── requirements.txt         # Dependencies
├── chunk_document.py        # Document preprocessing
├── embed_and_upload.py      # Embedding + upload to Pinecone
├── test_rag_system.py       # End-to-end tests
└── README.md               # This file
```

## Setup

### 1. Prerequisites

- Python 3.9+
- Pinecone account (free tier OK)
- Google AI Studio API key (Gemini)

### 2. Install Dependencies

```bash
# Activate virtual environment
./venv/Scripts/activate  # Windows
source venv/bin/activate  # Mac/Linux

# Install packages
pip install -r requirements.txt
```

### 3. Configure Environment

Create `.env` file:

```env
# Required
PINECONE_API_KEY=your_pinecone_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here

# Optional (defaults shown)
PINECONE_INDEX_NAME=hr-policy-rag
LLM_MODEL=gemini-2.0-flash-exp
EMBEDDING_MODEL=all-MiniLM-L6-v2
RETRIEVAL_TOP_K=5
RETRIEVAL_ALPHA=0.5
MEMORY_MAX_PAIRS=5
SESSION_TIMEOUT_MINUTES=30
API_PORT=8000
```

### 4. Prepare Data (Already Done)

If you need to reprocess the document:

```bash
# Step 1: Chunk document
python chunk_document.py

# Step 2: Generate embeddings and upload to Pinecone
python embed_and_upload.py
```

## Running the System

### Option 1: Run Tests First (Recommended)

```bash
python test_rag_system.py
```

This will test:
- Configuration validation
- Retriever (hybrid search)
- Generator (LLM + memory)
- RAG pipeline (end-to-end)
- API imports

### Option 2: Run API Server

```bash
python -m src.api.main
```

Or using uvicorn directly:

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Option 3: Test Individual Components

```bash
# Test retriever only
python -m src.core.retriever

# Test generator only
python -m src.core.generator

# Test RAG pipeline only
python -m src.core.rag
```

## Usage

### Web UI

1. Start server: `python -m src.api.main`
2. Open browser: http://localhost:8000
3. Ask questions!

Example questions:
- "How many vacation days do I get per year?"
- "What is the policy on overtime?"
- "Can I work from home?"

### API Endpoints

**Ask Question**
```bash
curl -X POST http://localhost:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the vacation policy?",
    "session_id": null,
    "top_k": 5,
    "use_memory": true
  }'
```

**Create Session**
```bash
curl -X POST http://localhost:8000/api/session/new
```

**Clear History**
```bash
curl -X POST http://localhost:8000/api/session/{session_id}/clear
```

**Get Configuration**
```bash
curl http://localhost:8000/api/config
```

**Health Check**
```bash
curl http://localhost:8000/api/health
```

## Configuration

All settings in `config/settings.py` can be overridden via environment variables.

### Key Parameters

**Retrieval**
- `RETRIEVAL_TOP_K` (1-20): Number of documents to retrieve
- `RETRIEVAL_ALPHA` (0-1): Hybrid search balance
  - 0.0 = pure keyword (BM25)
  - 0.5 = balanced (default)
  - 1.0 = pure semantic

**LLM**
- `LLM_TEMPERATURE` (0-2): Response creativity (0.3 = balanced)
- `LLM_MAX_OUTPUT_TOKENS`: Max response length (1024)

**Memory**
- `MEMORY_MAX_PAIRS`: Last N Q&A pairs to remember (5)
- `SESSION_TIMEOUT_MINUTES`: Session expiration (30)

## Memory System

### Simple Buffer Window

- Stores last 5 Q&A pairs per session
- In-memory (no persistence)
- Separate memory per session
- Automatic cleanup on session timeout

### How It Works

```python
# Session 1
User: "How many vacation days?"
Assistant: "15 days per year"

User: "Can I carry them over?"  # Has context from previous Q&A
Assistant: "Yes, up to 5 days"

# Memory: [Q1, A1, Q2, A2] → Used in next query
```

## Development

### Adding New Features

1. **Modify retrieval**: Edit `src/core/retriever.py`
2. **Modify generation**: Edit `src/core/generator.py`
3. **Add API endpoints**: Edit `src/api/main.py`
4. **Update UI**: Edit `frontend/` files

### Testing Changes

```bash
# Test specific component
python -m src.core.retriever

# Test full system
python test_rag_system.py
```

### Deployment

For production:

1. Set `API_RELOAD=false` in `.env`
2. Use production ASGI server (e.g., Gunicorn)
3. Add authentication middleware
4. Enable HTTPS
5. Set up monitoring (health check endpoint)

## Troubleshooting

### "PINECONE_API_KEY not set"
- Check `.env` file exists and contains valid API key
- Ensure `.env` is in project root directory

### "Index 'hr-policy-rag' does not exist"
- Run `python embed_and_upload.py` to create index
- Or change `PINECONE_INDEX_NAME` in `.env`

### "No results returned"
- Check Pinecone index has vectors: Run test script
- Verify query embeddings are generated correctly

### "Generator failed"
- Check GEMINI_API_KEY is valid
- Verify internet connection
- Check Gemini API quota/limits

### Session timeout
- Increase `SESSION_TIMEOUT_MINUTES` in `.env`
- Or create new session manually

## Performance

**Typical Response Times** (on standard hardware):
- Retrieval: 0.3-0.5s
- Generation: 1-2s
- **Total**: 1.5-2.5s

**Optimization Tips**:
- Reduce `RETRIEVAL_TOP_K` for faster retrieval
- Lower `LLM_MAX_OUTPUT_TOKENS` for faster generation
- Use `RETRIEVAL_ALPHA=1.0` for pure semantic (faster)

## Technical Interview Points

### Chunking Strategy
- **LangChain RecursiveCharacterTextSplitter**: AI-based semantic chunking
- **Hierarchical separators**: Paragraph → Sentence → Word → Character
- **Chunk size**: 1500 chars (~400 words) with 200 char overlap
- **Rich metadata**: Page numbers, sections, key terms extracted

### Hybrid Search
- **Dense vectors**: sentence-transformers (semantic similarity)
- **Sparse vectors**: BM25 via pinecone-text (keyword matching)
- **Server-side fusion**: Pinecone handles merging automatically
- **Why dotproduct metric**: Required for hybrid search with L2-normalized vectors

### Memory Design
- **Buffer window**: Simple FIFO queue (last 5 Q&A pairs)
- **Session-based**: Isolated memory per user session
- **No persistence**: In-memory only (stateless on restart)
- **Rationale**: Simple, predictable, no database needed

### Why These Choices?
- **Gemini 2.0 Flash**: Fast, cost-effective, good quality
- **Pinecone native hybrid**: No manual ranking, optimized fusion
- **Simple memory**: Easy to understand, debug, and maintain
- **No reranking**: 103 chunks is small, hybrid search sufficient

## License

MIT

## Support

For issues, check:
1. Test results: `python test_rag_system.py`
2. Logs: Check console output
3. Health endpoint: http://localhost:8000/api/health
