"""
Pydantic models and schemas for the healthcare backend API.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from datetime import datetime
from enum import Enum


class ActionType(str, Enum):
    """Enum for different action types."""

    GENERATE_GRAPH = "generate_graph"
    GENERATE_OFFER_LETTER = "generate_offer_letter"
    GENERATE_RELIEVING_LETTER = "generate_relieving_letter"
    REGULAR_QUERY = "regular_query"


class ModelType(str, Enum):
    """Enum for different model types."""

    RESUME_ANALYZER = "Resume Analyzer"
    PAYROLL_ANALYZER = "Payroll Analyzer"


class ChatEntry(BaseModel):
    """Model for individual chat entries."""

    id: str = Field(..., description="Unique identifier for the chat entry")
    question: str = Field(..., description="User's question or prompt")
    answer: str = Field(..., description="AI's response")
    graph: List[str] = Field(
        default_factory=list, description="S3 URLs for generated graphs"
    )
    generated_offer_letter: List[str] = Field(
        default_factory=list, description="S3 URLs for generated offer letters"
    )
    citation: List[str] = Field(default_factory=list, description="Source citations")
    created_at: Optional[datetime] = Field(
        default_factory=datetime.utcnow, description="Creation timestamp"
    )

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class Session(BaseModel):
    """Model for chat sessions."""

    data: List[ChatEntry] = Field(
        default_factory=list, description="List of chat entries"
    )
    updated_at: str = Field(..., description="Last update timestamp")
    created_at: str = Field(..., description="Creation timestamp")
    model_name: str = Field(default="", description="Model name used in this session")
    session_id: str = Field(..., description="Unique session identifier")
    session_base_identifier: Optional[str] = Field(
        None, description="Base identifier for the session"
    )
    title: Optional[str] = Field(
        None, description="Auto-generated title for the session based on user queries"
    )

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class CreateSessionRequest(BaseModel):
    """Request model for creating a new session."""

    pass


class ChatHistoryResponse(BaseModel):
    """Response model for chat history."""

    session_name: str = Field(..., description="Session name/ID")
    session: Session = Field(..., description="Session data")


class SessionListResponse(BaseModel):
    """Response model for listing all sessions."""

    sessions: Dict[str, Session] = Field(..., description="Dictionary of all sessions")


class PromptRequest(BaseModel):
    """Request model for prompts."""

    prompt: str = Field(
        ..., description="User's prompt or question", min_length=1, max_length=2000
    )


class UploadResponse(BaseModel):
    """Response model for file uploads."""

    info: str = Field(..., description="Upload information")
    file_path: Optional[str] = Field(None, description="Path where file was saved")
    s3_url: Optional[str] = Field(None, description="S3 URL if uploaded to cloud")


class QueryResponse(BaseModel):
    """Response model for query results."""

    answer: str = Field(..., description="AI's response")
    id: str = Field(..., description="Chat entry ID")
    citation: Optional[List[str]] = Field(None, description="Source citations")
    graph: Optional[List[str]] = Field(None, description="Generated graph URLs")
    generated_offer_letter: Optional[List[str]] = Field(
        None, description="Generated document URLs"
    )


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str = Field(..., description="Health status")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Check timestamp"
    )
    version: str = Field(..., description="Application version")

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class ErrorResponse(BaseModel):
    """Response model for errors."""

    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Error timestamp"
    )

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class RelievingLetterDetails(BaseModel):
    """Model for relieving letter details."""

    full_name: str = Field(..., description="Employee's full name")
    last_designation: str = Field(..., description="Last designation")
    first_date_of_employment: str = Field(..., description="First date of employment")
    last_working_date: str = Field(..., description="Last working date")
    designation: Optional[str] = Field(None, description="Current designation")


class RelievingLetterResponse(BaseModel):
    """Response model for relieving letter generation."""

    success: bool = Field(..., description="Whether the operation was successful")
    data: Optional[RelievingLetterDetails] = Field(
        None, description="Extracted details"
    )
    error: Optional[str] = Field(None, description="Error message if failed")
    s3_url: Optional[str] = Field(None, description="S3 URL of generated document")
