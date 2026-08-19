from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from inspect import signature

from browser_use import Agent, Browser, ChatOpenAI
from openai import pydantic_function_tool
from openai.types.responses import (
    FunctionToolParam,
    ResponseInputText,
)

from llm_utils.models.model_utils import get_model_def
from llm_utils.models.tool_models import (
    BrowserUseArgs,
    FuncCallStatus,
    TimeArgs,
    ToolOutput,
    WeatherArgs,
)


def get_tool_name(tool: Callable[..., Awaitable[ToolOutput]]) -> str:
    assert hasattr(tool, "__name__")
    assert isinstance(tool.__name__, str)
    tool_name: str = tool.__name__
    return tool_name


def get_tool_name_dict(
    tools: list[Callable[..., Awaitable[ToolOutput]]],
) -> dict[str, Callable[..., Awaitable[ToolOutput]]]:
    tools_dict: dict[str, Callable[..., Awaitable[ToolOutput]]] = {}
    for tool in tools:
        tool_name = get_tool_name(tool)
        tools_dict[tool_name] = tool
    return tools_dict


def get_tool_json_list(
    tools: list[Callable[..., Awaitable[ToolOutput]]],
) -> list[FunctionToolParam]:
    tools_json_list: list[FunctionToolParam] = []
    for tool in tools:
        tool_name = get_tool_name(tool)
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
    return tools_json_list


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
                ResponseInputText(
                    type="input_text",
                    text="This should contain details or instruction on what human input to give. This part will also be shown to user",
                ),
                ResponseInputText(type="input_text", text=result.model_dump_json()),
            ],
        )
    return ToolOutput(
        result=[
            ResponseInputText(
                type="input_text", text=f"The weather in {args.location} is Sunny"
            )
        ]
    )


async def get_time(args: TimeArgs) -> ToolOutput:
    return ToolOutput(
        result=[
            ResponseInputText(type="input_text", text=datetime.now(tz=UTC).isoformat())
        ]
    )


TOOLS_DICT = get_tool_name_dict([use_browser, get_weather, get_time])
