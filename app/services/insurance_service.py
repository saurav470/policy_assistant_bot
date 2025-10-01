"""
Insurance service for healthcare chatbot functionality.
"""

import json
import logging
import re
from typing import Dict, Any, Optional, List
from datetime import datetime
import asyncio

from openai import AsyncOpenAI
from app.config import settings
from app.models.insurance_schemas import (
    MobileExtractionResponse,
    PolicyInfo,
    InsuranceSession,
    InsuranceChatResponse,
    ContextAnalysisResponse,
    QueryTranslationResponse,
)

from app.utils.common import extract_pattern
from app.services.llm_services.models import GPT_4_1_MINI, GPT_4_1
from qdrant_client import models
from app.services.session_service import SessionService

# Import ingestion pipeline for RAG search
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from ingestion import HealthcareDocumentIngestion

logger = logging.getLogger(__name__)

session_service = SessionService()



class InsuranceService:
    """Service for insurance chatbot operations."""

    def __init__(self):
        """Initialize the insurance service."""
        self.openai_client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.policies_data: Dict[str, Any] = {}
        self.sessions: Dict[str, InsuranceSession] = {}
        self._load_policies_data()

        # Initialize Qdrant ingestion pipeline for RAG search
        try:
            self.ingestion_pipeline = HealthcareDocumentIngestion(
                qdrant_url=settings.qdrant_url,
                collection_name=settings.qdrant_collection_name,
                openai_api_key=settings.openai_api_key,
            )
            logger.info("Qdrant ingestion pipeline initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant ingestion pipeline: {e}")
            self.ingestion_pipeline = None

    def _load_policies_data(self) -> None:
        """Load policies data from JSON file."""
        try:
            with open("Insurance/purchase_policies.json", "r", encoding="utf-8") as f:
                self.policies_data = json.load(f)
            logger.info(
                f"Loaded {len(self.policies_data)} policies from purchase_policies.json"
            )
        except Exception as e:
            logger.error(f"Failed to load policies data: {str(e)}")
            self.policies_data = {}

    async def analyze_context_for_rag(
        self, prompt: str, policy_data: Optional[PolicyInfo] = None
    ) -> ContextAnalysisResponse:
        """
        Analyze user prompt to determine if purchase_policies.json is sufficient or if RAG search is needed.

        Args:
            prompt (str): User's prompt/question
            policy_data (Optional[PolicyInfo]): Policy data if available

        Returns:
            ContextAnalysisResponse: Analysis result with UIN numbers if RAG search needed
        """
        try:
            # Build context about available data
            available_data_context = ""
            uin_numbers = []

            if policy_data:
                available_data_context = f"""
                Available Policy Data:
                - Policy Number: {policy_data.policy_number}
                - Provider: {policy_data.policy_summary.get('provider', 'N/A')}
                - UIN: {policy_data.policy_summary.get('uin', 'N/A')}
                - Sum Insured: {policy_data.sum_insured}
                - Policy Period: {policy_data.policy_period.get('start', 'N/A')} to {policy_data.policy_period.get('end', 'N/A')}
                - Benefits: Basic hospitalization, pre/post hospitalization, ambulance, etc.
                - Premium: {policy_data.premium.get('amount', 'N/A')}
                """

                # Extract UIN for potential RAG search
                uin = policy_data.policy_summary.get("uin", "")
                if uin:
                    uin_numbers.append(uin)
            else:
                available_data_context = "No specific policy data available. Only general purchase_policies.json data is accessible."

            available_data_json = policy_data.model_dump_json()

            analysis_prompt = f"""
            <task>
            Analyze the user's query to determine if the available purchase_policies.json data is sufficient to answer their question, or if additional RAG search is needed for detailed policy documents.
            </task>
            
            <user_query>
            {prompt}
            </user_query>
            
            <available_data>
            {available_data_json}
            
            </available_data>
            
            <data_categories>
            Purchase Policies JSON contains:
            - Basic policy information (policy number, provider, UIN)
            - Policy holder details (name, age, contact)
            - Sum insured and premium information
            - Basic benefits overview (hospitalization, pre/post hospitalization, ambulance)
            - Policy period and renewal information
            - Contact information and agent details
            
            RAG Search would be needed for:
            - Detailed policy terms and conditions
            - Specific coverage exclusions and limitations
            - Detailed claim procedures and requirements
            - Specific waiting periods for different conditions
            - Detailed benefit descriptions and sub-limits
            - Policy wordings and legal clauses
            - Specific disease coverage details
            - Network hospital information
            - Detailed add-on benefits
            </data_categories>
            
            <instructions>
            1. Determine if the user's query can be answered with basic policy information from purchase_policies.json
            2. If the query requires detailed policy terms, exclusions, or specific coverage details, RAG search is needed
            3. If RAG search is needed, provide the UIN number(s) to search for
            4. Provide clear reasoning for your decision
            </instructions>
            
            <response_format>
            <is_purchase_policy_sufficient>true_or_false</is_purchase_policy_sufficient>
            <needs_rag_search>true_or_false</needs_rag_search>
            <uin_numbers>comma_separated_uin_numbers_or_empty</uin_numbers>
            <reasoning>explanation_of_decision</reasoning>
            </response_format>
            """

            response = await self.openai_client.chat.completions.create(
                model=GPT_4_1_MINI,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert insurance analyst. Analyze queries to determine if basic policy data is sufficient or if detailed policy documents are needed.",
                    },
                    {"role": "user", "content": analysis_prompt},
                ],
                temperature=0.1,
                max_tokens=300,
            )

            response_text = response.choices[0].message.content.strip()
            print("prompt for context analysis===", analysis_prompt)
            print("===Context Analysis Response===", response_text)

            # Extract structured response
            is_sufficient = extract_pattern(
                response_text, "is_purchase_policy_sufficient"
            )
            needs_rag = extract_pattern(response_text, "needs_rag_search")
            uin_list = extract_pattern(response_text, "uin_numbers")
            reasoning = extract_pattern(response_text, "reasoning")

            # Parse boolean values
            is_sufficient_bool = is_sufficient and is_sufficient.lower() == "true"
            needs_rag_bool = needs_rag and needs_rag.lower() == "true"

            # Parse UIN numbers
            uin_numbers_list = []
            if uin_list and uin_list.lower() != "empty" and uin_list.strip():
                uin_numbers_list = [
                    uin.strip() for uin in uin_list.split(",") if uin.strip()
                ]

            return ContextAnalysisResponse(
                is_purchase_policy_sufficient=is_sufficient_bool,
                needs_rag_search=needs_rag_bool,
                uin_numbers=uin_numbers_list,
                reasoning=reasoning or "Analysis completed",
            )

        except Exception as e:
            logger.error(f"Error analyzing context for RAG: {str(e)}")
            return ContextAnalysisResponse(
                is_purchase_policy_sufficient=True,
                needs_rag_search=False,
                uin_numbers=[],
                reasoning=f"Error in analysis: {str(e)}",
            )

    async def translate_query_for_rag(
        self, original_query: str, policy_data: Optional[PolicyInfo] = None
    ) -> QueryTranslationResponse:
        """
        Translate user query into multiple optimized queries for RAG search.

        Args:
            original_query (str): Original user query
            policy_data (Optional[PolicyInfo]): Policy data if available

        Returns:
            QueryTranslationResponse: Multiple translated queries optimized for RAG search
        """
        try:
            # Build policy context for better query translation
            policy_context = ""
            if policy_data:
                policy_context = f"""
                Policy Context:
                - Provider: {policy_data.policy_summary.get('provider', 'N/A')}
                - Policy Type: {policy_data.policy_summary.get('name', 'N/A')}
                - UIN: {policy_data.policy_summary.get('uin', 'N/A')}
                - Coverage Type: {policy_data.people.get('type', 'N/A')}
                """
                
            translation_prompt = f"""
            <task>
            Translate the user's insurance query into multiple optimized queries for RAG search. 
            Create different variations that will help retrieve the most relevant information from policy documents.
            </task>
            
            <original_query>
            {original_query}
            </original_query>
            
            {policy_context}
            
            <translation_guidelines>
            1. Create 3-5 different query variations
            2. Use different terminology and synonyms
            3. Include both specific and general versions
            4. Consider legal/technical terms vs. everyday language
            5. Include policy-specific terms when relevant
            6. Prioritize queries from most specific to most general
            7. Use insurance domain terminology
            </translation_guidelines>
            
            <query_types>
            - specific: Exact technical/legal terms
            - general: Common language equivalents
            - technical: Industry-specific terminology
            - legal: Policy document language
            - benefit: Coverage and benefit focused
            - procedural: Process and procedure focused
            </query_types>
            
            <examples>
            Original: "What's covered for hospitalization?"
            Translated:
            1. "hospitalization coverage benefits inpatient care" (specific)
            2. "what medical expenses are covered in hospital" (general)
            3. "inpatient treatment coverage policy terms" (technical)
            4. "hospitalization benefits and exclusions" (benefit)
            
            Original: "How to make a claim?"
            Translated:
            1. "claim procedure process steps" (procedural)
            2. "how to file insurance claim" (general)
            3. "claim submission requirements documentation" (technical)
            4. "claim process policy terms" (legal)
            </examples>
            
            <instructions>
            1. Generate 3-5 optimized queries
            2. Assign appropriate query types
            3. Set priority order (1=highest priority)
            4. Use keywords that would appear in policy documents
            5. Include both broad and narrow search terms
            </instructions>
            
            <response_format>
            <query_1>first_optimized_query</query_1>
            <type_1>query_type_1</type_1>
            <priority_1>1</priority_1>
            
            <query_2>second_optimized_query</query_2>
            <type_2>query_type_2</type_2>
            <priority_2>2</priority_2>
            
            <query_3>third_optimized_query</query_3>
            <type_3>query_type_3</type_3>
            <priority_3>3</priority_3>
            
            <query_4>fourth_optimized_query</query_4>
            <type_4>query_type_4</type_4>
            <priority_4>4</priority_4>
            
            <query_5>fifth_optimized_query</query_5>
            <type_5>query_type_5</type_5>
            <priority_5>5</priority_5>
            </response_format>
            """

            response = await self.openai_client.chat.completions.create(
                model=GPT_4_1_MINI,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert in insurance document search and query optimization. Create multiple query variations to maximize information retrieval from policy documents.",
                    },
                    {"role": "user", "content": translation_prompt},
                ],
                temperature=0.2,
                max_tokens=800,
            )

            response_text = response.choices[0].message.content.strip()
            print("===Query Translation Response===", response_text)

            # Extract translated queries
            queries = []
            types = []
            priorities = []

            for i in range(1, 6):  # Extract up to 5 queries
                query = extract_pattern(response_text, f"query_{i}")
                query_type = extract_pattern(response_text, f"type_{i}")
                priority = extract_pattern(response_text, f"priority_{i}")

                if query and query.strip():
                    queries.append(query.strip())
                    types.append(query_type.strip() if query_type else "general")
                    try:
                        priorities.append(int(priority) if priority else i)
                    except ValueError:
                        priorities.append(i)

            # Ensure we have at least one query
            if not queries:
                queries = [original_query]
                types = ["general"]
                priorities = [1]

            return QueryTranslationResponse(
                original_query=original_query,
                translated_queries=queries,
                query_types=types,
                search_priority=priorities,
            )

        except Exception as e:
            logger.error(f"Error translating query for RAG: {str(e)}")
            return QueryTranslationResponse(
                original_query=original_query,
                translated_queries=[original_query],
                query_types=["general"],
                search_priority=[1],
            )

    async def analyze_and_translate_query(
        self, prompt: str, policy_data: Optional[PolicyInfo] = None
    ) -> tuple[ContextAnalysisResponse, QueryTranslationResponse]:
        """
        Combined function to analyze context and translate query for RAG search.

        Args:
            prompt (str): User's prompt/question
            policy_data (Optional[PolicyInfo]): Policy data if available

        Returns:
            tuple: (ContextAnalysisResponse, QueryTranslationResponse)
        """
        try:
            # Run both analyses in parallel for efficiency
            context_analysis_task = self.analyze_context_for_rag(prompt, policy_data)
            query_translation_task = self.translate_query_for_rag(prompt, policy_data)

            context_analysis, query_translation = await asyncio.gather(
                context_analysis_task, query_translation_task
            )

            return context_analysis, query_translation

        except Exception as e:
            logger.error(f"Error in combined analysis and translation: {str(e)}")
            # Return fallback responses
            fallback_context = ContextAnalysisResponse(
                is_purchase_policy_sufficient=True,
                needs_rag_search=False,
                uin_numbers=[],
                reasoning=f"Error in analysis: {str(e)}",
            )
            fallback_translation = QueryTranslationResponse(
                original_query=prompt,
                translated_queries=[prompt],
                query_types=["general"],
                search_priority=[1],
            )
            return fallback_context, fallback_translation

    async def extract_mobile_number(self, prompt: str) -> MobileExtractionResponse:
        try:

            extraction_prompt = f"""
            <task>
            Extract the mobile number from the user's message. Look for Indian mobile numbers (10 digits starting with 6, 7, 8, or 9).
            </task>
            
            <user_message>
            {prompt}
            </user_message>
            
            <instructions>
            1. Look for 10-digit numbers that could be mobile numbers
            2. Consider numbers with country code (+91) or without
            3. Return the most likely mobile number
            4. If no mobile number found, return null
            </instructions>            
            """

            user_prompt = f""" 
            <response_format>
            <mobile_number>extracted_number_or_null</mobile_number>
            </response_format>
            """

            response = await self.openai_client.chat.completions.create(
                model=GPT_4_1_MINI,
                messages=[
                    {
                        "role": "system",
                        "content": extraction_prompt,
                    },
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                max_tokens=200,
            )

            response_text = response.choices[0].message.content.strip()

            print("===res form openai for phone number extraction===", response_text)

            mobile_number = extract_pattern(response_text, "mobile_number")
            print("===mobile number extracted===", mobile_number)
            message = ""

            if not mobile_number or mobile_number.lower() == "null":
                mobile_number = None
                
            # if moblie number search in purchase policies data
            if mobile_number:
                policy_data = self.find_policy_by_mobile(mobile_number)
                if policy_data:
                    print(
                        f"Mobile number {mobile_number} matched with policy {policy_data.policy_number}"
                    )
                else:
                    print(f"Mobile number {mobile_number} not found in any policy data")
                    message = f"I couldn't find any insurance policy associated with the mobile number ending with {mobile_number}. Please check the number or contact customer support for assistance."
                    mobile_number = None

            return MobileExtractionResponse(
                mobile_number=mobile_number,
                found=bool(mobile_number),
                message=(
                        "I couldn't find a mobile number in your message. "
                        "Please provide your mobile number so I can help you with your insurance policy information."
                        if not mobile_number and not message
                        else message
                    )
            )

        except Exception as e:
            logger.error(f"Error extracting mobile number: {str(e)}")
            return MobileExtractionResponse(
                mobile_number=None,
                found=bool(None),
            )

    # TODO: use graph database for better search
    def find_policy_by_mobile(self, mobile_number: str) -> Optional[PolicyInfo]:
        """
        Find policy information by mobile number.

        Args:
            mobile_number (str): Mobile number to search for

        Returns:
            Optional[PolicyInfo]: Policy information if found
        """
        try:
            # Search through all policies
            for policy_id, policy_data in self.policies_data.items():
                # Check policy holder mobile
                policy_holder = policy_data.get("people", {}).get("policy_holder", {})
                contact = policy_holder.get("contact", {})
                policy_mobile = contact.get("mobile", "")
                print("===policy mobile===", len(policy_mobile) , policy_mobile)
                # Extract last 4 digits for comparison (since numbers are masked)
                if policy_mobile and mobile_number:
                    policy_last_4 = (
                        policy_mobile[-4:] if len(policy_mobile) >= 4 else ""
                    )
                    mobile_last_4 = (
                        mobile_number[-4:] if len(mobile_number) >= 4 else ""
                    )

                    if policy_last_4 == mobile_last_4:
                        return PolicyInfo(
                            policy_number=policy_data.get("policy_summary", {}).get(
                                "policy_number", ""
                            ),
                            policy_summary=policy_data.get("policy_summary", {}),
                            people=policy_data.get("people", {}),
                            sum_insured=str(policy_data.get("sum_insured", "")),
                            policy_period=policy_data.get("policy_period", {}),
                            premium=policy_data.get("premium", {}),
                            benefits=policy_data.get("benefits", {}),
                            contact_information=policy_data.get(
                                "contact_information", {}
                            ),
                        )

            return None

        except Exception as e:
            logger.error(f"Error finding policy by mobile: {str(e)}")
            return None

    async def generate_ai_response(
        self, prompt: str, policy_data: Optional[PolicyInfo], mobile_number: str
    ) -> str:
        """
        Generate AI response using OpenAI with policy context and RAG search results.

        Args:
            prompt (str): User's prompt
            policy_data (Optional[PolicyInfo]): Policy information if found
            mobile_number (str): Mobile number

        Returns:
            str: AI generated response
        """
        try:
            # Analyze context and translate query for RAG search
            context_analysis, query_translation = (
                await self.analyze_and_translate_query(prompt, policy_data)
            )

            # Build base context from purchase policy data
            purchase_policy_context = self._build_context(policy_data, mobile_number)

            # Perform RAG search if needed
            rag_results = []
            rag_context = ""

            if context_analysis.needs_rag_search and self.ingestion_pipeline:
                logger.info("Performing RAG search for detailed policy information")

                # Get UIN numbers for filtering
                uin_numbers = context_analysis.uin_numbers
                if not uin_numbers and policy_data:
                    # Extract UIN from policy data if not provided in analysis
                    uin = policy_data.policy_summary.get("uin", "")
                    if uin:
                        uin_numbers = [uin]

                # Perform RAG search with optimized queries
                rag_results = await self.perform_rag_search(
                    queries=query_translation.translated_queries[
                        :3
                    ],  # Use top 3 queries
                    uin_numbers=uin_numbers,
                    limit_per_query=2,  # Limit results per query to avoid overwhelming
                )

                # Format RAG results for prompt
                if rag_results:
                    rag_context = self.format_rag_results_for_prompt(rag_results)
                    logger.info(
                        f"RAG search found {len(rag_results)} relevant documents"
                    )
                else:
                    rag_context = (
                        "No additional policy documents found for the specific query."
                    )
                    logger.info("RAG search returned no results")

            # Build comprehensive prompt with both purchase policy and RAG data
            ai_prompt = f"""
            <task>
            You are a helpful healthcare insurance chatbot. Respond to the user's query about their insurance policy using both basic policy information and detailed policy documents.
            </task>
            
            <user_query>
            {prompt}
            </user_query>
            
            <purchase_policy_information>
            {policy_data.model_dump_json()}
            </purchase_policy_information>
            
            <detailed_policy_documents>
            {rag_context}
            </detailed_policy_documents>
            
            <rag_analysis>
            Purchase Policy Sufficient: {context_analysis.is_purchase_policy_sufficient}
            Needs RAG Search: {context_analysis.needs_rag_search}
            UIN Numbers Searched: {', '.join(context_analysis.uin_numbers) if context_analysis.uin_numbers else 'None'}
            Reasoning: {context_analysis.reasoning}
            Queries Used: {', '.join(query_translation.translated_queries[:3])}
            </rag_analysis>
            
            <instructions>
            1. Use the purchase policy information for basic details (policy number, holder, sum insured, etc.)
            2. Use the detailed policy documents for specific coverage details, exclusions, procedures, etc.
            3. If both sources have information, prioritize the detailed policy documents for specific questions
            4. If no policy found, politely inform them and suggest contacting customer service
            5. Be helpful, professional, and concise
            6. Provide specific information from the policy documents when available
            7. If asked about benefits, coverage, or claims, provide relevant details from both sources
            8. Always cite the source of information (basic policy info vs. detailed policy documents)
            9. If information is not available in either source, suggest contacting customer service
            </instructions>
            
            <response_format>
            <Html>formatted_response_in_html</Html>
            </response_format>
            """

            print("===AI Prompt===", ai_prompt)

            response = await self.openai_client.chat.completions.create(
                model=GPT_4_1,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a professional healthcare insurance chatbot. Provide helpful and accurate information about insurance policies using both basic policy data and detailed policy documents.",
                    },
                    {"role": "user", "content": ai_prompt},
                ],
                temperature=0.3,
                max_tokens=800,  # Increased for more detailed responses
            )

            print("===AI Response===", response.choices[0].message.content.strip())
            res = extract_pattern(
                response.choices[0].message.content.strip(), "Html"
            )
            if res:
                return res.strip()
            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"Error generating AI response: {str(e)}")
            return "I apologize, but I'm having trouble processing your request right now. Please try again later or contact our customer service."

    def _build_context(
        self, policy_data: Optional[PolicyInfo], mobile_number: str
    ) -> str:
        """Build context string for AI response."""
        if not policy_data:
            return f"No policy found for mobile number ending in {mobile_number[-4:]}"

        context = f"""
        Policy Number: {policy_data.policy_number}
        Policy Holder: {policy_data.people.get('policy_holder', {}).get('name', 'N/A')}
        Sum Insured: {policy_data.sum_insured}
        Policy Period: {policy_data.policy_period.get('start', 'N/A')} to {policy_data.policy_period.get('end', 'N/A')}
        Premium: {policy_data.premium.get('amount', 'N/A')}
        Coverage Type: {policy_data.people.get('type', 'N/A')}
        """

        # Add benefits summary
        benefits = policy_data.benefits
        if benefits:
            context += f"""
        Benefits:
        - Hospitalization: {benefits.get('hospitalization', {}).get('coverage', 'N/A')}
        - Pre-hospitalization: {benefits.get('hospitalization', {}).get('pre_hospitalization_days', 'N/A')} days
        - Post-hospitalization: {benefits.get('hospitalization', {}).get('post_hospitalization_days', 'N/A')} days
        - Ambulance: {benefits.get('ambulance', 'N/A')}
        """

        return context

    def get_or_create_session(self, mobile_number: str) -> InsuranceSession:
        """Get existing session or create new one."""
        if mobile_number not in self.sessions:
            self.sessions[mobile_number] = InsuranceSession(
                mobile_number=mobile_number,
                policy_data=self.find_policy_by_mobile(mobile_number),
            )
        return self.sessions[mobile_number]

    def add_to_session_history(
        self, mobile_number: str, user_prompt: str, ai_response: str
    ) -> None:
        """Add conversation to session history."""
        session = self.get_or_create_session(mobile_number)

        chat_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "user_prompt": user_prompt,
            "ai_response": ai_response,
        }

        session.chat_history.append(chat_entry)
        session.updated_at = datetime.utcnow()

    def get_session_history(self, mobile_number: str) -> List[Dict[str, Any]]:
        """Get chat history for a session."""
        session = self.get_or_create_session(mobile_number)
        return session.chat_history

    def get_session_count(self) -> int:
        """Get total number of active sessions."""
        return len(self.sessions)

    def get_all_uin_numbers(self) -> List[str]:
        """Get all UIN numbers from policies data for RAG search."""
        uin_numbers = []
        try:
            for policy_id, policy_data in self.policies_data.items():
                uin = policy_data.get("policy_summary", {}).get("uin", "")
                if uin and uin not in uin_numbers:
                    uin_numbers.append(uin)
        except Exception as e:
            logger.error(f"Error extracting UIN numbers: {str(e)}")
        return uin_numbers

    def get_prioritized_queries(
        self, query_translation: QueryTranslationResponse
    ) -> List[Dict[str, Any]]:
        """
        Get translated queries sorted by priority for RAG search.

        Args:
            query_translation (QueryTranslationResponse): Query translation result

        Returns:
            List[Dict[str, Any]]: Prioritized queries with metadata
        """
        try:
            # Create list of query dictionaries with priority
            queries_with_priority = []
            for i, (query, query_type, priority) in enumerate(
                zip(
                    query_translation.translated_queries,
                    query_translation.query_types,
                    query_translation.search_priority,
                )
            ):
                queries_with_priority.append(
                    {
                        "query": query,
                        "type": query_type,
                        "priority": priority,
                        "index": i,
                    }
                )

            # Sort by priority (1 = highest priority)
            queries_with_priority.sort(key=lambda x: x["priority"])

            return queries_with_priority

        except Exception as e:
            logger.error(f"Error prioritizing queries: {str(e)}")
            return [
                {
                    "query": query_translation.original_query,
                    "type": "general",
                    "priority": 1,
                    "index": 0,
                }
            ]

    async def perform_rag_search(
        self, queries: List[str], uin_numbers: List[str], limit_per_query: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Perform RAG search using Qdrant vector database.
        """
        if not self.ingestion_pipeline:
            logger.warning("Qdrant ingestion pipeline not available for RAG search")
            return []

        try:
            all_results = []
            seen_documents = set()

            for query in queries:
                logger.info(f"Performing RAG search for query: {query}")

                if uin_numbers:
                    # Loop through each UIN number individually
                    for uin in uin_numbers:
                        logger.info(f"Searching for UIN: {uin}")

                        # Create filter for single UIN
                        filter_metadata = models.Filter(
                            must=[
                                models.FieldCondition(
                                    key="metadata.uin",
                                    match=models.MatchValue(value=uin),
                                )
                            ]
                        )

                        # Perform search for this specific UIN
                        results = self.ingestion_pipeline.search_documents(
                            query=query,
                            limit=5,
                            filter_metadata=filter_metadata,
                        )

                        logger.info(
                            f"Found {len(results)} results for UIN: {uin}  {results}"
                        )

                        # Add results to collection, avoiding duplicates
                        for result in results:
                            # Create unique identifier for document
                            doc_id = f"{result['metadata'].get('uin', 'unknown')}_{result['metadata'].get('chunk_index', 0)}"
                            if doc_id not in seen_documents:
                                seen_documents.add(doc_id)
                                all_results.append(
                                    {
                                        "content": result["content"],
                                        "metadata": result["metadata"],
                                        "similarity_score": result["similarity_score"],
                                        "query_used": query,
                                        "uin": result["metadata"].get("uin", "unknown"),
                                        "searched_uin": uin,  # Track which UIN was searched
                                    }
                                )
                else:
                    # No UIN filter - search without filter
                    logger.info("No UIN filter provided, searching all documents")
                    results = self.ingestion_pipeline.search_documents(
                        query=query, limit=limit_per_query, filter_metadata=None
                    )

                    # Add results to collection
                    for result in results:
                        doc_id = f"{result['metadata'].get('uin', 'unknown')}_{result['metadata'].get('chunk_index', 0)}"
                        if doc_id not in seen_documents:
                            seen_documents.add(doc_id)
                            all_results.append(
                                {
                                    "content": result["content"],
                                    "metadata": result["metadata"],
                                    "similarity_score": result["similarity_score"],
                                    "query_used": query,
                                    "uin": result["metadata"].get("uin", "unknown"),
                                    "searched_uin": None,
                                }
                            )

            # Sort by similarity score (highest first)
            all_results.sort(key=lambda x: x["similarity_score"], reverse=True)
            logger.info(
                f"RAG search completed. Found {len(all_results)} unique documents"
            )
            return all_results

        except Exception as e:
            logger.error(f"Error performing RAG search: {str(e)}")
            return []

    def format_rag_results_for_prompt(self, rag_results: List[Dict[str, Any]]) -> str:
        """
        Format RAG search results for inclusion in GPT prompt.

        Args:
            rag_results: List of RAG search results

        Returns:
            Formatted string for GPT prompt
        """
        if not rag_results:
            return "No additional policy documents found."

        formatted_results = []
        for i, result in enumerate(rag_results, 1):
            metadata = result["metadata"]
            content = result["content"]
            similarity_score = result["similarity_score"]
            uin = result["uin"]

            formatted_result = f"""
Document {i} (UIN: {uin}, Relevance: {similarity_score:.3f}):
{content}

---"""
            formatted_results.append(formatted_result)

        return "\n".join(formatted_results)

    def is_rag_available(self) -> bool:
        """
        Check if RAG search is available.

        Returns:
            bool: True if RAG search is available, False otherwise
        """
        return self.ingestion_pipeline is not None

    def get_rag_status(self) -> Dict[str, Any]:
        """
        Get RAG system status information.

        Returns:
            Dict with RAG system status
        """
        status = {
            "rag_available": self.is_rag_available(),
            "qdrant_configured": False,
            "collection_exists": False,
            "total_documents": 0,
        }

        if self.ingestion_pipeline:
            try:
                # Check if Qdrant is accessible
                collections = self.ingestion_pipeline.qdrant_client.get_collections()
                status["qdrant_configured"] = True

                # Check if our collection exists
                collection_names = [col.name for col in collections.collections]
                if settings.qdrant_collection_name in collection_names:
                    status["collection_exists"] = True

                    # Get collection info
                    collection_info = (
                        self.ingestion_pipeline.qdrant_client.get_collection(
                            settings.qdrant_collection_name
                        )
                    )
                    status["total_documents"] = collection_info.points_count

            except Exception as e:
                logger.error(f"Error checking RAG status: {e}")
                status["error"] = str(e)

        return status


# Global insurance service instance
insurance_service = InsuranceService()
