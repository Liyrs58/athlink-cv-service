from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
from services.conversation_service import ask_conversation

router = APIRouter()


class ConversationRequest(BaseModel):
    question: str
    job_id: Optional[str] = None


@router.post("/conversation")
async def conversation(req: ConversationRequest):
    """
    Ask the Conversation Coach a question about your match data.

    The coach answers using ONLY stored match data — never generic advice.
    Optionally pass a job_id to ask about a specific match.
    """
    result = ask_conversation(question=req.question, job_id=req.job_id)
    return JSONResponse(result)
