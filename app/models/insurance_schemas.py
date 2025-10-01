"""
Pydantic models for healthcare insurance chatbot.
"""

from pydantic import BaseModel, Field, validator
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone


class InsurancePromptRequest(BaseModel):
    """Request model for insurance chatbot prompts."""

    prompt: str = Field(
        ..., description="User's prompt or question", min_length=1, max_length=2000
    )
    session_id: str = Field(..., description="Session identifier")
    mobile_number: Optional[str] = Field(
        None, description="Mobile number if already known"
    )


class MobileExtractionResponse(BaseModel):
    """Response model for mobile number extraction."""

    mobile_number: Optional[str] = Field(None, description="Extracted mobile number")
    found: bool = Field(False, description="Whether mobile number was found")
    message: Optional[str] = Field(None, description="Message regarding extraction")


class PolicyInfo(BaseModel):
    """Model for policy information."""

    policy_number: str = Field(..., description="Policy number")
    policy_summary: Dict[str, Any] = Field(
        ..., description="Policy summary information"
    )
    people: Dict[str, Any] = Field(..., description="People information")
    sum_insured: str = Field(..., description="Sum insured amount")
    policy_period: Dict[str, Any] = Field(..., description="Policy period details")
    premium: Dict[str, Any] = Field(..., description="Premium information")
    benefits: Dict[str, Any] = Field(..., description="Benefits information")
    contact_information: Dict[str, Any] = Field(..., description="Contact information")


class InsuranceChatResponse(BaseModel):
    """Response model for insurance chatbot."""

    mobile_number: str = Field(..., description="Mobile number used for lookup")
    policy_found: bool = Field(..., description="Whether policy was found")
    policy_data: Optional[PolicyInfo] = Field(
        None, description="Policy information if found"
    )
    ai_response: str = Field(..., description="AI generated response")
    session_id: str = Field(..., description="Session identifier ")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Response timestamp",
    )

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class InsuranceSession(BaseModel):
    """Model for insurance chat sessions using mobile number as identifier."""

    mobile_number: str = Field(..., description="Mobile number (session identifier)")
    chat_history: List[Dict[str, Any]] = Field(
        default_factory=list, description="Chat history"
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="Session creation time"
    )
    updated_at: datetime = Field(
        default_factory=datetime.utcnow, description="Last update time"
    )
    policy_data: Optional[PolicyInfo] = Field(
        None, description="Associated policy data"
    )
    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class QueryTranslationResponse(BaseModel):
    """Response model for query translation to generate multiple RAG-optimized queries."""

    original_query: str = Field(..., description="Original user query")
    translated_queries: List[str] = Field(
        ..., description="List of translated/optimized queries for RAG search"
    )
    query_types: List[str] = Field(
        ...,
        description="Types of queries (e.g., 'specific', 'general', 'technical', 'legal')",
    )
    search_priority: List[int] = Field(
        ..., description="Priority order for searching (1=highest priority)"
    )


class ContextAnalysisResponse(BaseModel):
    """Response model for context analysis to determine if RAG search is needed."""

    is_purchase_policy_sufficient: bool = Field(
        ...,
        description="Whether purchase_policies.json data is sufficient for the query",
    )
    needs_rag_search: bool = Field(
        ..., description="Whether additional RAG search is needed"
    )
    uin_numbers: List[str] = Field(
        default_factory=list, description="UIN numbers to search in RAG if needed"
    )
    reasoning: str = Field(
        ..., description="Explanation of why RAG search is or isn't needed"
    )


class InsuranceErrorResponse(BaseModel):
    """Error response model for insurance API."""

    error: str = Field(..., description="Error message")
    mobile_number: Optional[str] = Field(None, description="Mobile number if available")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Error timestamp"
    )

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}
