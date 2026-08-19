import asyncio
import json
import uuid
from collections.abc import AsyncIterable, Awaitable, Callable, Sequence
from datetime import UTC, datetime, timedelta
from enum import Enum
from functools import partial
from inspect import signature
from typing import Annotated, Any, Literal, TypedDict

import jwt
from annotated_types import Ge, Gt, Le, Lt, MaxLen, MinLen, MultipleOf
from browser_use import Agent, Browser, ChatOpenAI
from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.sse import EventSourceResponse, ServerSentEvent
from openai import AsyncOpenAI, pydantic_function_tool
from openai.types.responses import (
    EasyInputMessage,
    FunctionToolParam,
    ResponseFunctionToolCall,
    ResponseFunctionToolCallOutputItem,
    ResponseFunctionToolCallParam,
    ResponseInputItemParam,
    ResponseInputParam,
    ResponseInputText,
    ResponseOutputItem,
    ResponseOutputMessageParam,
    ResponseReasoningItemParam,
)
from openai.types.responses.response_custom_tool_call_output import (
    OutputOutputContentList,
)
from openai.types.responses.response_input_param import FunctionCallOutput
from pwdlib import PasswordHash
from pydantic import BaseModel, Field, Strict, create_model
from pydantic.fields import FieldInfo
from pydantic.json_schema import SkipJsonSchema
from sqlalchemy import select
from sqlalchemy.orm import (
    Session,
)


class FuncCallStatus(str, Enum):
    COMPLETED = "completed"
    INCOMPLETE = "incomplete"
    IN_PROGRESS = "in_progress"


class ToolOutput(BaseModel):
    status: FuncCallStatus = FuncCallStatus.COMPLETED
    result: list[OutputOutputContentList]


class ToolInput(BaseModel):
    human_input: SkipJsonSchema[bool] = Field(
        default=False, description="Whether the input was provided by Human or LLM"
    )


class BrowserUseArgs(ToolInput):
    __doc__ = "This function is for using BrowserUse"
    task: str = Field(description="Detailed instructions to pass to BrowserUse")


class WeatherArgs(ToolInput):
    __doc__ = "This function is for getting the weather for any location."
    location: str = Field(
        description="The location for which we want to know the weather"
    )
    feedback: SkipJsonSchema[str] = ""


class TimeArgs(ToolInput):
    __doc__ = "This function is for getting current UTC time in isoformat"
