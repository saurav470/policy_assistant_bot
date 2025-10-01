"""
Session management service for handling chat sessions.
"""

import logging
from typing import Dict, Optional, List
from datetime import datetime

from app.models.schemas import Session, ChatEntry
from app.config import settings
from app.global_store import singleton
from app.services.gemini_services import GeminiService


logger = logging.getLogger(__name__)


@singleton
class SessionService:
    """Service for managing chat sessions."""

    def __init__(self):
        """Initialize the session service."""
        self._sessions: Dict[str, Session] = {}
        self._max_sessions = settings.max_sessions_per_user
        self._session_timeout_hours = settings.session_timeout_hours
        self._gemini_service = GeminiService()

    def create_session(self, session_id: str, session: Session) -> bool:
        """
        Create a new session.

        Args:
            session_id (str): Unique session identifier
            session (Session): Session object to store

        Returns:
            bool: True if session was created successfully
        """
        try:
            # Check if we've reached the maximum number of sessions
            if len(self._sessions) >= self._max_sessions:
                self._cleanup_old_sessions()

            self._sessions[session_id] = session
            logger.info(f"Created new session: {session_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to create session {session_id}: {str(e)}")
            return False

    def get_session(self, session_id: str) -> Optional[Session]:
        """
        Get a session by ID.

        Args:
            session_id (str): Session identifier

        Returns:
            Optional[Session]: Session object if found, None otherwise
        """
        return self._sessions.get(session_id)

    def get_all_sessions(self) -> Dict[str, Session]:
        """
        Get all active sessions.

        Returns:
            Dict[str, Session]: Dictionary of all sessions
        """
        return self._sessions.copy()

    def update_session(self, session_id: str, session: Session) -> bool:
        """
        Update an existing session.

        Args:
            session_id (str): Session identifier
            session (Session): Updated session object

        Returns:
            bool: True if session was updated successfully
        """
        try:
            if session_id in self._sessions:
                session.updated_at = datetime.utcnow().isoformat() + "Z"
                self._sessions[session_id] = session
                logger.info(f"Updated session: {session_id}")
                return True
            return False

        except Exception as e:
            logger.error(f"Failed to update session {session_id}: {str(e)}")
            return False

    def update_session_base_identifier(
        self, session_id: str, base_identifier: str
    ) -> bool:
        """
        Update the session base identifier for a given session.

        Args:
            session_id (str): Session identifier
            base_identifier (str): New base identifier value

        Returns:
            bool: True if session base identifier was updated successfully
        """
        try:
            if session_id in self._sessions:
                session = self._sessions[session_id]
                session.session_base_identifier = base_identifier
                session.updated_at = datetime.utcnow().isoformat() + "Z"
                logger.info(
                    f"Updated session base identifier for session: {session_id}"
                )
                return True
            return False

        except Exception as e:
            logger.error(
                f"Failed to update base identifier for session {session_id}: {str(e)}"
            )
            return False

    def update_session(
        self, session_id: str, **updates
    ) -> bool:
        """
        Update fields for a given session.

        Args:
            session_id (str): Session identifier
            updates: key-value pairs of fields to update

        Returns:
            bool: True if updated successfully
        """
        try:
            if session_id not in self._sessions:
                return False

            session = self._sessions[session_id]

            for field, value in updates.items():
                if hasattr(session, field):
                    setattr(session, field, value)

            session.updated_at = datetime.utcnow().isoformat() + "Z"
            logger.info(f"Updated session {session_id} with {updates}")
            return True

        except Exception as e:
            logger.error(f"Failed to update session {session_id}: {str(e)}")
            return False


    def add_chat_entry(self, session_id: str, chat_entry: ChatEntry) -> bool:
        """
        Add a chat entry to a session.

        Args:
            session_id (str): Session identifier
            chat_entry (ChatEntry): Chat entry to add

        Returns:
            bool: True if entry was added successfully
        """
        try:
            if session_id in self._sessions:
                session = self._sessions[session_id]
                session.data.append(chat_entry)
                session.updated_at = datetime.utcnow().isoformat() + "Z"
                logger.info(f"Added chat entry to session: {session_id}")
                return True
            return False

        except Exception as e:
            logger.error(f"Failed to add chat entry to session {session_id}: {str(e)}")
            return False

    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session.

        Args:
            session_id (str): Session identifier

        Returns:
            bool: True if session was deleted successfully
        """
        try:
            if session_id in self._sessions:
                del self._sessions[session_id]
                logger.info(f"Deleted session: {session_id}")
                return True
            return False

        except Exception as e:
            logger.error(f"Failed to delete session {session_id}: {str(e)}")
            return False

    def clear_session_history(self, session_id: str) -> bool:
        """
        Clear chat history for a session.

        Args:
            session_id (str): Session identifier

        Returns:
            bool: True if history was cleared successfully
        """
        try:
            if session_id in self._sessions:
                session = self._sessions[session_id]
                session.data = []
                session.updated_at = datetime.utcnow().isoformat() + "Z"
                logger.info(f"Cleared history for session: {session_id}")
                return True
            return False

        except Exception as e:
            logger.error(f"Failed to clear history for session {session_id}: {str(e)}")
            return False

    def _cleanup_old_sessions(self) -> None:
        """
        Clean up old sessions based on timeout.
        """
        try:
            current_time = datetime.utcnow()
            sessions_to_remove = []

            for session_id, session in self._sessions.items():
                try:
                    session_time = datetime.fromisoformat(
                        session.updated_at.replace("Z", "+00:00")
                    )
                    time_diff = current_time - session_time.replace(tzinfo=None)

                    if time_diff.total_seconds() > (self._session_timeout_hours * 3600):
                        sessions_to_remove.append(session_id)

                except Exception as e:
                    logger.warning(
                        f"Error parsing session time for {session_id}: {str(e)}"
                    )
                    sessions_to_remove.append(session_id)

            for session_id in sessions_to_remove:
                del self._sessions[session_id]
                logger.info(f"Cleaned up old session: {session_id}")

        except Exception as e:
            logger.error(f"Error during session cleanup: {str(e)}")

    def get_session_count(self) -> int:
        """
        Get the total number of active sessions.

        Returns:
            int: Number of active sessions
        """
        return len(self._sessions)

    def is_session_valid(self, session_id: str) -> bool:
        """
        Check if a session is valid and not expired.

        Args:
            session_id (str): Session identifier

        Returns:
            bool: True if session is valid
        """
        session = self.get_session(session_id)
        if not session:
            return False

        try:
            session_time = datetime.fromisoformat(
                session.updated_at.replace("Z", "+00:00")
            )
            current_time = datetime.utcnow()
            time_diff = current_time - session_time.replace(tzinfo=None)

            return time_diff.total_seconds() < (self._session_timeout_hours * 3600)

        except Exception as e:
            logger.warning(f"Error validating session {session_id}: {str(e)}")
            return False

    async def generate_and_set_title(self, session_id: str) -> Optional[str]:
        """
        Generate and set a title for the session based on user queries.
        
        Args:
            session_id (str): Session identifier
            
        Returns:
            Optional[str]: Generated title or None if failed
        """
        try:
            session = self.get_session(session_id)
            if not session or not session.data:
                return None
            
            # Extract user queries (questions) from chat entries
            user_queries = [entry.question for entry in session.data if entry.question.strip()]
            
            if len(user_queries) < 1:
                return None
                
            # Generate title using Gemini
            title = await self._gemini_service.generate_session_title(user_queries)
            
            if title:
                # Update session with the generated title
                session.title = title
                session.updated_at = datetime.utcnow().isoformat() + "Z"
                logger.info(f"Generated and set title '{title}' for session: {session_id}")
                return title
                
            return None
            
        except Exception as e:
            logger.error(f"Failed to generate title for session {session_id}: {str(e)}")
            return None

    def should_generate_title(self, session_id: str) -> bool:
        """
        Check if a title should be generated for this session.
        Title is generated after 3 user queries and only if not already set.
        
        Args:
            session_id (str): Session identifier
            
        Returns:
            bool: True if title should be generated
        """
        try:
            session = self.get_session(session_id)
            if not session:
                return False
                
            # Don't generate if title already exists
            if session.title:
                return False
                
            # Generate title after 3 user queries
            user_queries = [entry.question for entry in session.data if entry.question.strip()]
            return len(user_queries) >= 3
            
        except Exception as e:
            logger.error(f"Error checking title generation for session {session_id}: {str(e)}")
            return False
