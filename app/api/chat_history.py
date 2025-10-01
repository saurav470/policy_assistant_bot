"""
Chat history API endpoints for managing conversation sessions.
"""

import uuid
from datetime import datetime
from typing import Dict
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse

from app.models.schemas import (
    CreateSessionRequest,
    ChatHistoryResponse,
    SessionListResponse,
    Session,
    ChatEntry,
)
from app.services.session_service import SessionService

# Create router
router = APIRouter(prefix="/api/v1", tags=["chat-history"])

# Initialize session service
session_service = SessionService()


@router.post("/sessions", response_model=dict)
async def create_session():
    """
    Create a new chat session.

    Returns:
        dict: Session creation response with session_id
    """
    try:
        session_id = str(uuid.uuid4())
        current_time = datetime.utcnow().isoformat() + "Z"

        session = Session(
            data=[],
            updated_at=current_time,
            created_at=current_time,
            model_name="",
            session_id=session_id,
        )

        session_service.create_session(session_id, session)

        return {
            "message": "Session created successfully",
            "session_id": session_id,
            "created_at": current_time,
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create session: {str(e)}",
        )


@router.get("/sessions/{session_id}", response_model=ChatHistoryResponse)
async def get_chat_history(session_id: str):
    """
    Get chat history for a specific session.

    Args:
        session_id (str): The session identifier

    Returns:
        ChatHistoryResponse: Session data with chat history

    Raises:
        HTTPException: If session not found
    """
    try:
        session = session_service.get_session(session_id)
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Session not found"
            )

        return ChatHistoryResponse(session_name=session_id, session=session)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve chat history: {str(e)}",
        )


@router.get("/sessions", response_model=SessionListResponse)
async def list_sessions():
    """
    List all active sessions.

    Returns:
        SessionListResponse: Dictionary of all sessions
    """
    try:
        sessions = session_service.get_all_sessions()
        return SessionListResponse(sessions=sessions)

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list sessions: {str(e)}",
        )


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """
    Delete a specific session.

    Args:
        session_id (str): The session identifier

    Returns:
        dict: Deletion confirmation

    Raises:
        HTTPException: If session not found
    """
    try:
        success = session_service.delete_session(session_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Session not found"
            )

        return {"message": "Session deleted successfully", "session_id": session_id}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete session: {str(e)}",
        )


@router.put("/sessions/{session_id}/clear")
async def clear_session_history(session_id: str):
    """
    Clear chat history for a specific session.

    Args:
        session_id (str): The session identifier

    Returns:
        dict: Clear confirmation

    Raises:
        HTTPException: If session not found
    """
    try:
        success = session_service.clear_session_history(session_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Session not found"
            )

        return {
            "message": "Session history cleared successfully",
            "session_id": session_id,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear session history: {str(e)}",
        )


@router.get("/sessions/{session_id}/stats")
async def get_session_stats(session_id: str):
    """
    Get statistics for a specific session.

    Args:
        session_id (str): The session identifier

    Returns:
        dict: Session statistics

    Raises:
        HTTPException: If session not found
    """
    try:
        session = session_service.get_session(session_id)
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Session not found"
            )

        stats = {
            "session_id": session_id,
            "total_messages": len(session.data),
            "created_at": session.created_at,
            "updated_at": session.updated_at,
            "model_name": session.model_name,
            "has_graphs": any(entry.graph for entry in session.data),
            "has_documents": any(
                entry.generated_offer_letter for entry in session.data
            ),
            "has_citations": any(entry.citation for entry in session.data),
        }

        return stats

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get session stats: {str(e)}",
        )
