import asyncio
from typing import Any, Literal

from openai.types.responses import (
    ResponseFunctionToolCall,
    ResponseFunctionToolCallOutputItem,
)
from pydantic import BaseModel


class HumanInputRequired(BaseModel):
    type: Literal["human_input_required"]
    tool_call_param: ResponseFunctionToolCall
    tool_call_output: ResponseFunctionToolCallOutputItem


class HumanInputFromUser(BaseModel):
    type: Literal["human_input_from_user"]
    call_id: str
    human_input: dict[str, Any]
