from enum import Enum

from openai.types.responses.response_custom_tool_call_output import (
    OutputOutputContentList,
)
from pydantic import BaseModel, Field
from pydantic.json_schema import SkipJsonSchema


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


class BaseA2AArgs(ToolInput):
    a2a_input: str = Field(description="The input text for A2A Agent")


def get_a2a_arg(doc: str):
    new_arg = BaseA2AArgs
    new_arg.__doc__ = doc
    return new_arg
