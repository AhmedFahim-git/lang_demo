import asyncio
import json
import uuid
from collections.abc import AsyncIterable, Awaitable, Callable
from datetime import UTC, datetime
from enum import Enum
from inspect import signature
from typing import Any, Literal, TypedDict

from annotated_types import Ge, Gt, Le, Lt, MaxLen, MinLen, MultipleOf
from browser_use import Agent, Browser, ChatOpenAI
from fastapi import FastAPI, HTTPException, status
from fastapi.sse import EventSourceResponse, ServerSentEvent
from openai import AsyncOpenAI, pydantic_function_tool
from openai.types.responses import (
    EasyInputMessage,
    EasyInputMessageParam,
    FunctionToolParam,
    ResponseFunctionToolCall,
    ResponseFunctionToolCallOutputItem,
    ResponseFunctionToolCallParam,
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
from pydantic import BaseModel, Field, Strict, create_model
from pydantic.fields import FieldInfo
from pydantic.json_schema import SkipJsonSchema

client = AsyncOpenAI(api_key="None", base_url="http://localhost:8080/v1")


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


class HumanInputRequired(BaseModel):
    type: Literal["human_input_required"]
    tool_call_param: ResponseFunctionToolCall
    tool_call_output: ResponseFunctionToolCallOutputItem


class HumanInputFromUser(BaseModel):
    type: Literal["human_input_from_user"]
    tool_call_param: ResponseFunctionToolCall
    tool_call_output: ResponseFunctionToolCallOutputItem
    human_input: dict[str, Any]


class FieldParams(TypedDict, total=False):
    description: str
    default: str | int | float | bool
    strict: bool
    gt: int | float
    ge: int | float
    lt: int | float
    le: int | float
    multiple_of: int | float
    allow_inf_nan: bool
    min_length: int
    max_length: int


STR_TYPE_MAP: dict[str, type] = {
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
}


TYPE_STR_MAP: dict[type, str] = {v: k for k, v in STR_TYPE_MAP.items()}

METADATA_TYPE_STR: dict[Any, str] = {
    Ge: "ge",
    Gt: "gt",
    Le: "le",
    Lt: "lt",
    MultipleOf: "multiple_of",
    MinLen: "min_length",
    MaxLen: "max_length",
    Strict: "strict",
}


class PydanticFieldDef(BaseModel):
    name: str
    type: str
    field_params: FieldParams


class PydanticModelDef(BaseModel):
    name: str
    fields: list[PydanticFieldDef]


def make_pydantic_model_from_def(model_def_str: str):
    model_def = PydanticModelDef.model_validate_json(model_def_str)
    fields: dict[str, Any] = {
        field_def.name: (STR_TYPE_MAP[field_def.type], Field(**field_def.field_params))
        for field_def in model_def.fields
    }
    return create_model(model_def.name, **fields)


class BrowserUseArgs(ToolInput):
    __doc__ = "This function is for using BrowserUse"
    task: str = Field(description="Detailed instructions to pass to BrowserUse")


async def use_browser(args: BrowserUseArgs) -> ToolOutput:
    browser_session = Browser(cdp_url="ws://127.0.0.1:3000/chromium?&timeout=300000")
    llm = ChatOpenAI(
        model="Qwen3.6", api_key="None", base_url="http://localhost:8080/v1"
    )
    agent = Agent(task=args.task, llm=llm, browser_session=browser_session)
    await agent.run()
    return ToolOutput(
        result=[ResponseInputText(text="Completed Successfully", type="input_text")]
    )


class WeatherArgs(ToolInput):
    __doc__ = "This function is for getting the weather for any location."
    location: str = Field(
        description="The location for which we want to know the weather"
    )
    feedback: SkipJsonSchema[str] = ""


def get_model_def(
    req_fields: dict[str, dict[str, str | int | float | bool | None]],
    fields_dict: dict[str, FieldInfo],
    name: str,
) -> PydanticModelDef:
    fields: list[PydanticFieldDef] = []
    for field, in_attrs in req_fields.items():
        field_info = fields_dict[field]
        field_info_dict = field_info.asdict()
        print(field_info_dict)
        attributes = field_info_dict.get("attributes")
        metadata = field_info_dict.get("metadata")
        metadata_dict: dict[str, str | int | float | bool] = {}
        for item in metadata:
            if type(item) in METADATA_TYPE_STR:
                metadata_str = METADATA_TYPE_STR[type(item)]
                metadata_dict[metadata_str] = getattr(item, metadata_str)
            elif hasattr(item, "allow_inf_nan"):
                metadata_dict["allow_inf_nan"] = item.allow_inf_nan

        if field_info.is_required():
            attributes["default"] = None
        attributes.update(metadata_dict)
        attributes.update(in_attrs)
        field_def: FieldParams = FieldParams(
            **{
                k: attributes[k]
                for k in FieldParams.__annotations__
                if attributes.get(k) is not None
            }
        )
        assert field_info.annotation is not None
        fields.append(
            PydanticFieldDef(
                name=field,
                type=TYPE_STR_MAP[field_info.annotation],
                field_params=field_def,
            )
        )
    return PydanticModelDef(name=name, fields=fields)


async def get_weather(args: WeatherArgs) -> ToolOutput:
    if not args.human_input:
        req_fields: dict[str, dict[str, str | int | float | bool | None]] = {
            "feedback": {"default": None},
        }
        result = get_model_def(
            req_fields=req_fields,
            fields_dict=args.__class__.model_fields,
            name="HumanInput",
        )
        return ToolOutput(
            status=FuncCallStatus.IN_PROGRESS,
            result=[
                ResponseInputText(type="input_text", text=result.model_dump_json())
            ],
        )
    return ToolOutput(
        result=[
            ResponseInputText(
                type="input_text", text=f"The weather in {args.location} is Sunny"
            )
        ]
    )


class TimeArgs(ToolInput):
    __doc__ = "This function is for getting current UTC time in isoformat"


async def get_time(args: TimeArgs) -> ToolOutput:
    return ToolOutput(
        result=[
            ResponseInputText(type="input_text", text=datetime.now(tz=UTC).isoformat())
        ]
    )


def get_tool_list_dict(
    tools: list[Callable[..., Awaitable[ToolOutput]]],
) -> tuple[list[FunctionToolParam], dict[str, Callable[..., Awaitable[ToolOutput]]]]:
    tools_json_list: list[FunctionToolParam] = []
    tools_dict: dict[str, Callable[..., Awaitable[ToolOutput]]] = {}
    for tool in tools:
        assert hasattr(tool, "__name__")
        assert isinstance(tool.__name__, str)
        tool_name: str = tool.__name__
        tool_annotation = signature(tool).parameters["args"].annotation
        tools_json_list.append(
            FunctionToolParam(
                type="function",
                name=tool_name,
                description=tool_annotation.__doc__,
                strict=True,
                parameters=pydantic_function_tool(tool_annotation)
                .get("function")
                .get("parameters"),
            )
        )
        tools_dict[tool_name] = tool
    return tools_json_list, tools_dict


tools_json_list, tools_dict = get_tool_list_dict(
    tools=[use_browser, get_weather, get_time]
)

# input_list: ResponseInputParam = [
#     EasyInputMessageParam(
#         content="What is weather in Amsterdam? What is the current UTC time",
#         role="user",
#     )
# ]


def append_output(
    response_output: list[ResponseOutputItem], input_list: ResponseInputParam
) -> None:
    for item in response_output:
        if item.type == "reasoning":
            input_list.append(ResponseReasoningItemParam(item.model_dump()))
        elif item.type == "message":
            input_list.append(ResponseOutputMessageParam(item.model_dump()))
        elif item.type == "function_call":
            input_list.append(ResponseFunctionToolCallParam(item.model_dump()))


async def process_tool_call(
    tool_call: ResponseFunctionToolCall,
    tool: Callable[..., Awaitable[ToolOutput]],
) -> ResponseFunctionToolCallOutputItem:
    tool_annotation = signature(tool).parameters["args"].annotation
    result = await tool(tool_annotation(**json.loads(tool_call.arguments)))
    return ResponseFunctionToolCallOutputItem(
        call_id=tool_call.call_id,
        output=result.result,
        type="function_call_output",
        id=uuid.uuid4().hex,
        status=result.status.value,
    )


async def run_tool_call(
    tool_call: ResponseFunctionToolCall,
    tools_dict: dict[str, Callable[..., Awaitable[ToolOutput]]],
    input_list: ResponseInputParam,
    pending_call_ids: set[str],
) -> list[ServerSentEvent]:
    res = await process_tool_call(tool_call=tool_call, tool=tools_dict[tool_call.name])
    call_output = res.output
    assert isinstance(call_output, list)
    res_list: list[ServerSentEvent] = []
    for item in call_output:
        if item.type == "input_file":
            res_list.append(ServerSentEvent(data=item))
    if res.status == "completed":
        input_list.append(FunctionCallOutput(res.model_dump()))
    elif res.status == "in_progress":
        pending_call_ids.add(tool_call.call_id)
        res_list.append(
            ServerSentEvent(
                data=HumanInputRequired(
                    type="human_input_required",
                    tool_call_param=tool_call,
                    tool_call_output=res,
                )
            )
        )
    return res_list


app = FastAPI()


input_list: ResponseInputParam = []
pending_call_ids: set[str] = set()


async def run_model_loop(
    input_list: ResponseInputParam, pending_call_ids: set[str]
) -> AsyncIterable[ServerSentEvent]:
    if pending_call_ids:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="We should not be create model response while tool call human input is pending",
        )
    tool_calls: list[ResponseFunctionToolCall] = []
    while True:
        response = await client.responses.create(
            instructions="You are a helpful agent",
            tools=tools_json_list,
            input=input_list,
        )
        append_output(response_output=response.output, input_list=input_list)
        for item in response.output:
            if item.type == "function_call":
                tool_calls.append(item)
            elif item.type == "message":
                yield ServerSentEvent(data=item)
        if not tool_calls:
            print(f"reached case with no tool call: {input_list}")
            break
        tasks = [
            asyncio.create_task(
                run_tool_call(
                    tool_call=tool_call,
                    tools_dict=tools_dict,
                    input_list=input_list,
                    pending_call_ids=pending_call_ids,
                )
            )
            for tool_call in tool_calls
        ]
        for task in asyncio.as_completed(tasks):
            res = await task
            for item in res:
                yield item
        if pending_call_ids:
            break
        tool_calls = []
    print(f"Out of loop: {input_list}")


# Need to break up this function and remove print statements
@app.post("/", response_class=EventSourceResponse)
async def run_model(
    model_input: EasyInputMessage | HumanInputFromUser,
) -> AsyncIterable[ServerSentEvent]:
    print(model_input)
    if isinstance(model_input, EasyInputMessage):
        if pending_call_ids:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="We should not be getting input while tool call human input is pending",
            )
        input_list.append(EasyInputMessageParam(model_input.model_dump()))
        print(input_list)
        # Pending_call_ids is modified in place
        async for sse_event in run_model_loop(
            input_list=input_list, pending_call_ids=pending_call_ids
        ):
            yield sse_event
    elif isinstance(model_input, HumanInputFromUser):
        if model_input.tool_call_param.call_id not in pending_call_ids:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="This function call has already been processed",
            )
        arguments_json: dict[str, Any] = json.loads(
            model_input.tool_call_param.arguments
        )
        assert isinstance(model_input.tool_call_output.output[0], ResponseInputText)
        model_def = model_input.tool_call_output.output[0].text
        human_input_model = make_pydantic_model_from_def(model_def)
        human_input = human_input_model(**model_input.human_input)
        arguments_json.update(human_input.model_dump())
        arguments_json["human_input"] = True
        model_input.tool_call_param.arguments = json.dumps(arguments_json)
        try:
            pending_call_ids.remove(model_input.tool_call_param.call_id)
            for sse_event in await run_tool_call(
                tool_call=model_input.tool_call_param,
                tools_dict=tools_dict,
                input_list=input_list,
                pending_call_ids=pending_call_ids,
            ):
                yield sse_event
        except Exception as e:
            pending_call_ids.add(model_input.tool_call_param.call_id)
            print(e)
        if not pending_call_ids:
            async for sse_event in run_model_loop(
                input_list=input_list, pending_call_ids=pending_call_ids
            ):
                yield sse_event
