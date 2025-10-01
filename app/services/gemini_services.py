import os
import httpx
import logging
from typing import List, Optional
from app.config import settings

logger = logging.getLogger(__name__)


class GeminiService:
    """Service for interacting with Gemini 2.0 Flash model."""

    def __init__(self):
        self.api_key= settings.gemini_api_key
        if not self.api_key:
            raise ValueError("Gemini API key is missing.")

        self.endpoint = (
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent"
            f"?key={self.api_key}"
        )

    async def generate_session_title(self, user_queries: List[str]) -> Optional[str]:
        """
        Generate a 2-3 word title based on the first 3 user queries.
        
        Args:
            user_queries: List of user queries from the session
            
        Returns:
            str: Generated title or None if generation fails
        """
        try:
            # Take only the first 3 queries for title generation
            top_queries = user_queries[:3]
            
            if not top_queries:
                return None
                
            queries_text = "\n".join([f"- {query}" for query in top_queries])
            
            prompt = f"""Based on the following user queries, generate a concise 2-3 word title that summarizes the main topic:

{queries_text}

Instructions:
- Generate only 2-3 words
- No quotes or punctuation
- Capture the main topic/theme
- Examples: "Policy Benefits", "Claim Process", "Coverage Details"

Title:"""

            payload = {
                "contents": [
                    {
                        "parts": [{"text": prompt}]
                    }
                ],
                "generationConfig": {
                    "temperature": 0.3,
                    "topK": 32,
                    "topP": 1,
                    "maxOutputTokens": 10,
                    "stopSequences": []
                }
            }

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(self.endpoint, json=payload)
                response.raise_for_status()
                result = response.json()

                if "candidates" in result and len(result["candidates"]) > 0:
                    candidate = result["candidates"][0]
                    if "content" in candidate and "parts" in candidate["content"]:
                        title = candidate["content"]["parts"][0]["text"].strip()
                        # Clean up the title - remove quotes, extra spaces, etc.
                        title = title.replace('"', '').replace("'", '').strip()
                        # Limit to 3 words max
                        words = title.split()
                        if len(words) > 3:
                            title = " ".join(words[:3])
                        return title
                
                logger.warning("No valid response from Gemini for title generation")
                return None

        except Exception as e:
            logger.error(f"Error generating session title: {str(e)}")
            return None
