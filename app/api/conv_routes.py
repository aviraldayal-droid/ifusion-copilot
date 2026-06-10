"""
Conversation history endpoints for TBG Copilot.

All endpoints require authentication.

GET    /api/v1/conversations/                      — list user's conversations
POST   /api/v1/conversations/                      — create conversation
GET    /api/v1/conversations/{conv_id}/messages    — list messages
DELETE /api/v1/conversations/{conv_id}             — delete conversation (204)
PATCH  /api/v1/conversations/{conv_id}/title       — update title
"""
from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.auth.deps import get_current_user
from app.db.auth_store import (
    create_conversation,
    delete_conversation,
    get_conversation,
    list_conversations,
    list_messages,
    update_conversation_title,
)

log = logging.getLogger("tbg.conv")
router = APIRouter(prefix="/api/v1/conversations", tags=["conversations"])


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class CreateConvRequest(BaseModel):
    title: str = Field(default="New Chat", max_length=500)
    mode:  str = Field(default="db",       max_length=20)


class UpdateTitleRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=500)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _require_ownership(conv_id: int, user_id: int) -> dict:
    """Load the conversation and verify it belongs to user_id; raise 403/404 otherwise."""
    conv = await asyncio.to_thread(get_conversation, conv_id)
    if conv is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found.")
    if conv["user_id"] != user_id:
        log.warning("ownership check failed: conv_id=%s user_id=%s owner_id=%s", conv_id, user_id, conv["user_id"])
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied.")
    return conv


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/")
async def list_user_conversations(current_user: dict = Depends(get_current_user)):
    """Return all conversations for the authenticated user, newest first."""
    convs = await asyncio.to_thread(list_conversations, current_user["id"])
    log.debug("list_conversations: user_id=%s count=%d", current_user["id"], len(convs))
    return {"conversations": convs}


@router.post("/", status_code=status.HTTP_201_CREATED)
async def create_user_conversation(
    body: CreateConvRequest,
    current_user: dict = Depends(get_current_user),
):
    """Create a new conversation and return it."""
    conv = await asyncio.to_thread(
        create_conversation, current_user["id"], body.title, body.mode
    )
    log.info("create_conversation: id=%s user_id=%s mode=%s title=%r", conv["id"], current_user["id"], body.mode, body.title)
    return conv


@router.get("/{conv_id}/messages")
async def get_conversation_messages(
    conv_id: int,
    current_user: dict = Depends(get_current_user),
):
    """Return all messages in a conversation (must be owned by the caller)."""
    await _require_ownership(conv_id, current_user["id"])
    msgs = await asyncio.to_thread(list_messages, conv_id)
    log.debug("get_messages: conv_id=%s user_id=%s count=%d", conv_id, current_user["id"], len(msgs))
    return {"conversation_id": conv_id, "messages": msgs}


@router.delete("/{conv_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user_conversation(
    conv_id: int,
    current_user: dict = Depends(get_current_user),
):
    """Delete a conversation.  Returns 404 if not found or not owned by caller."""
    deleted = await asyncio.to_thread(delete_conversation, conv_id, current_user["id"])
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found.")
    log.info("delete_conversation: conv_id=%s user_id=%s", conv_id, current_user["id"])


@router.patch("/{conv_id}/title")
async def update_title(
    conv_id: int,
    body: UpdateTitleRequest,
    current_user: dict = Depends(get_current_user),
):
    """Update the title of a conversation."""
    await _require_ownership(conv_id, current_user["id"])
    await asyncio.to_thread(update_conversation_title, conv_id, body.title)
    log.info("update_title: conv_id=%s user_id=%s title=%r", conv_id, current_user["id"], body.title)
    return {"conv_id": conv_id, "title": body.title}
