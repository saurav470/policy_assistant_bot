"""
Healthcare Insurance Chatbot API endpoints.
"""

import logging
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse

from app.models.insurance_schemas import (
    InsurancePromptRequest,
    InsuranceChatResponse,
    InsuranceErrorResponse,
)
from app.services.insurance_service import insurance_service
from datetime import datetime, timezone
from app.services.session_service import SessionService
from app.models.schemas import ChatEntry


# Create router
router = APIRouter(prefix="/api/v1/insurance", tags=["insurance-chatbot"])

logger = logging.getLogger(__name__)

session_service = SessionService()


# @router.post("/chat", response_model=InsuranceChatResponse)
# async def insurance_chatbot(request: InsurancePromptRequest):
#     try:
#         logger.info(f"Received insurance chatbot request: {request.prompt[:100]}...")

#         is_session_valid = session_service.is_session_valid(request.session_id)
#         if not is_session_valid:
#             return JSONResponse(
#                 status_code=status.HTTP_400_BAD_REQUEST,
#                 content={
#                     "error": "Invalid or expired session",
#                     "detail": "The provided session ID is invalid or has expired. Please create a new session.",
#                     "session_id": request.session_id,
#                 },
#             )
#         current_state_session = session_service.get_session(request.session_id)
#         # Step 1: Extract mobile number
#         mobile_number = current_state_session.session_base_identifier

#         if not mobile_number:
#             mobile_extraction = await insurance_service.extract_mobile_number(
#                 request.prompt
#             )
#             mobile_number = mobile_extraction.mobile_number

#             if not mobile_number:
#                 # If no mobile number found, return error

#                 return InsuranceChatResponse(
#                     mobile_number="",
#                     policy_found=False,
#                     policy_data=None,
#                     ai_response="I couldn't find a mobile number in your message. Please provide your mobile number so I can help you with your insurance policy information.",
#                     session_id="",
#                 )
#             # Update session with extracted mobile number
#             res = InsuranceChatResponse(
#                 mobile_number=mobile_number,
#                 policy_found=False,
#                 policy_data=None,
#                 ai_response=f"Mobile number is registered successfully. How can I assist you further?",
#                 session_id=mobile_number,
#             )
#             session_service.update_session_base_identifier(
#                 request.session_id, mobile_number
#             )
#             session_service.update_session(
#                 request.session_id, data=[*current_state_session.data, res]
#             )
#             return res

#         logger.info(f"Using mobile number: {mobile_number[-4:]}****")

#         # Step 2: Find policy by mobile number
#         policy_data = insurance_service.find_policy_by_mobile(mobile_number)
#         policy_found = policy_data is not None

#         logger.info(f"Policy found: {policy_found}")

#         # Step 3: Generate AI response with policy context
#         ai_response = await insurance_service.generate_ai_response(
#             request.prompt, policy_data, mobile_number
#         )

#         # Step 4: Save to session history
#         insurance_service.add_to_session_history(
#             mobile_number, request.prompt, ai_response
#         )

#         # Step 5: Return response
#         response = InsuranceChatResponse(
#             mobile_number=mobile_number,
#             policy_found=policy_found,
#             policy_data=policy_data,
#             ai_response=ai_response,
#             session_id=mobile_number,  # Mobile number as session ID
#         )

#         logger.info(
#             f"Successfully processed insurance chatbot request for mobile: {mobile_number[-4:]}****"
#         )
#         return response

#     except Exception as e:
#         logger.error(f"Error in insurance chatbot: {str(e)}", exc_info=True)
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"Failed to process insurance chatbot request: {str(e)}",
#         )


from uuid import uuid4


@router.post("/chat", response_model=ChatEntry)
async def insurance_chatbot(request: InsurancePromptRequest):
    try:
        logger.info(f"Received insurance chatbot request: {request.prompt[:100]}...")

        # Step 0: validate session
        if not session_service.is_session_valid(request.session_id):
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "error": "Invalid or expired session",
                    "detail": "The provided session ID is invalid or has expired. Please create a new session.",
                    "session_id": request.session_id,
                },
            )

        current_state_session = session_service.get_session(request.session_id)
        mobile_number = current_state_session.session_base_identifier

        # Step 1: Extract mobile number if not already in session
        if not mobile_number:
            session_service.update_session(
                request.session_id, model_name="Policy Buddy"
            )
            mobile_extraction = await insurance_service.extract_mobile_number(
                request.prompt
            )
            mobile_number = mobile_extraction.mobile_number

            if not mobile_number:
                # No mobile number found
                chat_entry = ChatEntry(
                    id=str(uuid4()),
                    question=request.prompt,
                    answer=mobile_extraction.message
                    or "I couldn't find a mobile number in your message. Please provide your mobile number so, I can help you with your insurance policy information.",
                    graph=[],
                    generated_offer_letter=[],
                    citation=[],
                )
                session_service.update_session(
                    request.session_id, data=[*current_state_session.data, chat_entry]
                )
                
                # Generate title if needed
                try:
                    if session_service.should_generate_title(request.session_id):
                        await session_service.generate_and_set_title(request.session_id)
                except Exception as e:
                    logger.warning(f"Failed to generate title for session {request.session_id}: {str(e)}")
                
                return chat_entry

            # Mobile number found → update session base identifier
            session_service.update_session_base_identifier(
                request.session_id, mobile_number
            )

            # Build ChatEntry
            chat_entry = ChatEntry(
                id=str(uuid4()),
                question=request.prompt,
                answer=f"We found your mobile number and fetched your policy documents",
                graph=[],
                generated_offer_letter=[],
                citation=[],
            )

            # Save chat history
            session_service.update_session(
                request.session_id, data=[*current_state_session.data, chat_entry]
            )

            # Generate title if needed
            try:
                if session_service.should_generate_title(request.session_id):
                    await session_service.generate_and_set_title(request.session_id)
            except Exception as e:
                logger.warning(f"Failed to generate title for session {request.session_id}: {str(e)}")

            return chat_entry

        logger.info(f"Using mobile number: {mobile_number[-4:]}****")

        # Step 2: Find policy
        policy_data = insurance_service.find_policy_by_mobile(mobile_number)
        policy_found = policy_data is not None
        logger.info(f"Policy found: {policy_found}")

        # Step 3: Generate AI response
        ai_response = await insurance_service.generate_ai_response(
            request.prompt, policy_data, mobile_number
        )

        # Step 4: Build ChatEntry
        chat_entry = ChatEntry(
            id=str(uuid4()),
            question=request.prompt,
            answer=ai_response,
            graph=[],
            generated_offer_letter=[],
            citation=[],
        )

        # Step 5: Save chat history
        session_service.update_session(
            request.session_id, data=[*current_state_session.data, chat_entry]
        )

        # Step 6: Generate title if needed (after 3 user queries)
        try:
            if session_service.should_generate_title(request.session_id):
                await session_service.generate_and_set_title(request.session_id)
        except Exception as e:
            # Don't fail the request if title generation fails
            logger.warning(f"Failed to generate title for session {request.session_id}: {str(e)}")

        logger.info(
            f"Successfully processed insurance chatbot request for mobile: {mobile_number[-4:]}****"
        )
        return chat_entry

    except Exception as e:
        logger.error(f"Error in insurance chatbot: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process insurance chatbot request: {str(e)}",
        )
