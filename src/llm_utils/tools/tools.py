from collections.abc import Awaitable, Callable
from copy import copy
from datetime import UTC, datetime
from functools import partial
from inspect import signature

import httpx
from a2a.client import (
    A2ACardResolver,
    AuthInterceptor,
    ClientCallContext,
    ClientConfig,
    CredentialService,
    create_client,
)
from a2a.helpers import get_message_text, new_text_message
from a2a.types import Role, SendMessageRequest
from browser_use import Agent, Browser, ChatOpenAI
from openai import pydantic_function_tool
from openai.types.responses import (
    FunctionToolParam,
    ResponseInputText,
)
from sqlalchemy.orm import Session

from llm_utils.core.settings import settings
from llm_utils.db.db_utils import get_agents
from llm_utils.models.model_utils import get_model_def
from llm_utils.models.tool_models import (
    BaseA2AArgs,
    BrowserUseArgs,
    FuncCallStatus,
    TimeArgs,
    ToolOutput,
    WeatherArgs,
    get_a2a_arg,
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


class APICredentilService(CredentialService):
    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    async def get_credentials(
        self,
        security_scheme_name: str,
        context: ClientCallContext | None,
    ) -> str | None:
        if security_scheme_name == "apikey":
            return self.api_key
        else:
            return None


async def run_a2a_base(args: BaseA2AArgs, agent_id: int) -> ToolOutput:
    async with httpx.AsyncClient() as httpx_client:
        resolver = A2ACardResolver(
            httpx_client=httpx_client, base_url=f"{settings.a2a_base_url}/{agent_id}"
        )
        public_card = await resolver.get_agent_card()
    config = ClientConfig(streaming=True)
    interceptor = AuthInterceptor(APICredentilService(settings.a2a_api_key))
    client = await create_client(
        agent=public_card, client_config=config, interceptors=[interceptor]
    )
    message = new_text_message(text=args.a2a_input, role=Role.ROLE_USER)
    request = SendMessageRequest(message=message)
    async for chunk in client.send_message(request=request):
        if chunk.HasField("message"):
            return ToolOutput(
                result=[
                    ResponseInputText(
                        type="input_text", text=get_message_text(chunk.message)
                    )
                ]
            )
    return ToolOutput(result=[])


def get_tools_dict(
    db_session: Session | None = None,
) -> dict[str, Callable[..., Awaitable[ToolOutput]]]:
    tools_list = [use_browser, get_weather, get_time]
    if db_session is not None:
        agents = get_agents(db_session)
        for agent in agents:
            agent_func = partial(copy(run_a2a_base), agent_id=agent.agent_user_id)
            agent_func.__name__ = agent.agent_name
            # Even in ADK, agent card description is used
            agent_func.func.__annotations__["args"] = get_a2a_arg(
                agent.agent_description
            )
            tools_list.append(agent_func)

    return get_tool_name_dict(tools_list)
