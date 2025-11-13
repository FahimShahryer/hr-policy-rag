"""
FastAPI Backend for RAG System
Handles HTTP requests, session management, and CORS
"""

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import uuid
import time
from datetime import datetime, timedelta
from contextlib import asynccontextmanager

from src.core.rag import RAGPipeline
from config.settings import settings, validate_settings


# ========== Session Management ==========

class SessionManager:
    """
    Simple in-memory session management
    Each session has its own RAG pipeline instance with separate memory
    """

    def __init__(self, timeout_minutes: int = 30):
        """
        Initialize session manager

        Args:
            timeout_minutes: Session timeout in minutes
        """
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.timeout_minutes = timeout_minutes

    def create_session(self) -> str:
        """
        Create new session

        Returns:
            str: Session ID
        """
        session_id = str(uuid.uuid4())

        # Create new RAG pipeline for this session
        rag = RAGPipeline()
        rag.initialize()

        self.sessions[session_id] = {
            "rag": rag,
            "created_at": datetime.now(),
            "last_activity": datetime.now()
        }

        print(f"[INFO] Created session: {session_id}")
        return session_id

    def get_session(self, session_id: str) -> Optional[RAGPipeline]:
        """
        Get RAG pipeline for session

        Args:
            session_id: Session ID

        Returns:
            RAGPipeline if session exists and valid, None otherwise
        """
        if session_id not in self.sessions:
            return None

        session = self.sessions[session_id]

        # Check if session expired
        if self._is_expired(session):
            self.delete_session(session_id)
            return None

        # Update last activity
        session["last_activity"] = datetime.now()
        return session["rag"]

    def delete_session(self, session_id: str):
        """
        Delete session and cleanup resources

        Args:
            session_id: Session ID
        """
        if session_id in self.sessions:
            # Cleanup RAG pipeline
            self.sessions[session_id]["rag"].close()
            del self.sessions[session_id]
            print(f"[INFO] Deleted session: {session_id}")

    def _is_expired(self, session: Dict[str, Any]) -> bool:
        """
        Check if session is expired

        Args:
            session: Session dict

        Returns:
            bool: True if expired
        """
        timeout = timedelta(minutes=self.timeout_minutes)
        return (datetime.now() - session["last_activity"]) > timeout

    def cleanup_expired_sessions(self):
        """
        Remove all expired sessions
        """
        expired = [
            session_id
            for session_id, session in self.sessions.items()
            if self._is_expired(session)
        ]

        for session_id in expired:
            self.delete_session(session_id)

        if expired:
            print(f"[INFO] Cleaned up {len(expired)} expired sessions")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get session statistics

        Returns:
            Dict with session stats
        """
        return {
            "total_sessions": len(self.sessions),
            "timeout_minutes": self.timeout_minutes
        }


# ========== Global State ==========

# Session manager (initialized in lifespan)
session_manager: Optional[SessionManager] = None


# ========== Lifespan Management ==========

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup/shutdown
    """
    # Startup
    print("\n" + "=" * 80)
    print("  STARTING RAG API SERVER")
    print("=" * 80)
    print()

    # Validate configuration
    try:
        validate_settings()
        print("[OK] Configuration validated")
    except ValueError as e:
        print(f"[ERROR] Configuration invalid: {e}")
        raise

    # Initialize session manager
    global session_manager
    session_manager = SessionManager(timeout_minutes=settings.SESSION_TIMEOUT_MINUTES)
    print(f"[OK] Session manager initialized (timeout: {settings.SESSION_TIMEOUT_MINUTES}m)")

    print("\n" + "=" * 80)
    print("[SUCCESS] RAG API Server Ready!")
    print("=" * 80)
    print()

    yield

    # Shutdown
    print("\n[INFO] Shutting down RAG API server...")
    # Cleanup all sessions
    if session_manager:
        for session_id in list(session_manager.sessions.keys()):
            session_manager.delete_session(session_id)
    print("[INFO] Shutdown complete")


# ========== FastAPI App ==========

app = FastAPI(
    title="HR Policy RAG API",
    description="RAG-based chatbot for HR policy questions with conversation memory",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (frontend)
app.mount("/static", StaticFiles(directory="frontend"), name="static")


# ========== Request/Response Models ==========

class AskRequest(BaseModel):
    """Request model for /ask endpoint"""
    question: str = Field(..., min_length=1, description="User question")
    session_id: Optional[str] = Field(None, description="Session ID (optional, will create new if not provided)")
    top_k: Optional[int] = Field(None, ge=1, le=20, description="Number of documents to retrieve")
    use_memory: bool = Field(True, description="Whether to use conversation history")


class AskResponse(BaseModel):
    """Response model for /ask endpoint"""
    answer: str = Field(..., description="Generated answer")
    session_id: str = Field(..., description="Session ID")
    sources: List[Dict[str, Any]] = Field(..., description="Retrieved source documents")
    retrieval_time: float = Field(..., description="Retrieval time in seconds")
    generation_time: float = Field(..., description="Generation time in seconds")
    total_time: float = Field(..., description="Total time in seconds")
    num_sources: int = Field(..., description="Number of sources retrieved")
    success: bool = Field(..., description="Whether request succeeded")
    warning: Optional[str] = Field(None, description="Warning message if any")
    error: Optional[str] = Field(None, description="Error message if any")


class SessionResponse(BaseModel):
    """Response model for session endpoints"""
    session_id: str = Field(..., description="Session ID")
    message: str = Field(..., description="Status message")


class HealthResponse(BaseModel):
    """Response model for /health endpoint"""
    status: str = Field(..., description="Health status")
    timestamp: str = Field(..., description="Current timestamp")
    sessions: Dict[str, Any] = Field(..., description="Session statistics")


# ========== Helper Functions ==========

def get_session_manager() -> SessionManager:
    """Dependency to get session manager"""
    if session_manager is None:
        raise HTTPException(status_code=503, detail="Session manager not initialized")
    return session_manager


# ========== API Endpoints ==========

@app.get("/", response_class=HTMLResponse)
async def root():
    """
    Serve main chat UI
    """
    try:
        with open("frontend/index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(
            content="<h1>Frontend not found</h1><p>Please ensure frontend/index.html exists</p>",
            status_code=404
        )


@app.post("/api/ask", response_model=AskResponse)
async def ask_question(
    request: AskRequest,
    sm: SessionManager = Depends(get_session_manager)
):
    """
    Ask a question and get answer

    - Creates new session if session_id not provided
    - Uses existing session if valid session_id provided
    - Returns answer with sources and metadata
    """
    try:
        # Get or create session
        if request.session_id:
            rag = sm.get_session(request.session_id)
            if not rag:
                # Session expired or invalid, create new one
                session_id = sm.create_session()
                rag = sm.get_session(session_id)
            else:
                session_id = request.session_id
        else:
            # Create new session
            session_id = sm.create_session()
            rag = sm.get_session(session_id)

        # Ask question
        result = rag.ask(
            question=request.question,
            top_k=request.top_k,
            use_memory=request.use_memory
        )

        # Return response
        return AskResponse(
            session_id=session_id,
            **result
        )

    except Exception as e:
        print(f"[ERROR] /api/ask failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/session/new", response_model=SessionResponse)
async def create_new_session(sm: SessionManager = Depends(get_session_manager)):
    """
    Create new session
    """
    try:
        session_id = sm.create_session()
        return SessionResponse(
            session_id=session_id,
            message="New session created"
        )
    except Exception as e:
        print(f"[ERROR] /api/session/new failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/session/{session_id}/clear")
async def clear_session(
    session_id: str,
    sm: SessionManager = Depends(get_session_manager)
):
    """
    Clear conversation history for session
    """
    rag = sm.get_session(session_id)
    if not rag:
        raise HTTPException(status_code=404, detail="Session not found or expired")

    rag.clear_conversation()
    return {"message": "Conversation history cleared", "session_id": session_id}


@app.delete("/api/session/{session_id}")
async def delete_session(
    session_id: str,
    sm: SessionManager = Depends(get_session_manager)
):
    """
    Delete session
    """
    if session_id not in sm.sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    sm.delete_session(session_id)
    return {"message": "Session deleted", "session_id": session_id}


@app.get("/api/session/{session_id}/history")
async def get_history(
    session_id: str,
    sm: SessionManager = Depends(get_session_manager)
):
    """
    Get conversation history for session
    """
    rag = sm.get_session(session_id)
    if not rag:
        raise HTTPException(status_code=404, detail="Session not found or expired")

    history = rag.get_conversation_history()
    return {
        "session_id": session_id,
        "history": history,
        "num_messages": len(history)
    }


@app.get("/api/health", response_model=HealthResponse)
async def health_check(sm: SessionManager = Depends(get_session_manager)):
    """
    Health check endpoint
    """
    # Cleanup expired sessions
    sm.cleanup_expired_sessions()

    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        sessions=sm.get_stats()
    )


@app.get("/api/config")
async def get_config():
    """
    Get public configuration (non-sensitive values)
    """
    return {
        "retrieval_top_k": settings.RETRIEVAL_TOP_K,
        "retrieval_alpha": settings.RETRIEVAL_ALPHA,
        "memory_max_pairs": settings.MEMORY_MAX_PAIRS,
        "llm_model": settings.LLM_MODEL,
        "llm_temperature": settings.LLM_TEMPERATURE,
        "embedding_model": settings.EMBEDDING_MODEL
    }


# ========== Error Handlers ==========

@app.exception_handler(404)
async def not_found_handler(request: Request, exc: HTTPException):
    """Handle 404 errors"""
    return JSONResponse(
        status_code=404,
        content={"detail": "Endpoint not found"}
    )


@app.exception_handler(500)
async def internal_error_handler(request: Request, exc: Exception):
    """Handle 500 errors"""
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


# ========== Main ==========

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.API_RELOAD
    )
