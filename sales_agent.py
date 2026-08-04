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

from .database import (
    HumanUserModel,
    MessageModel,
    SessionModel,
    UserModel,
    get_db_session,
)

client = AsyncOpenAI(api_key="None", base_url="http://localhost:8080/v1")

# Generated using "openssl rand -hex 32". Move this to a .env
SECRET_KEY = "bd121e4ec165595a80f1cd5da97e80318fe0c0484c24739697c037aab9bd04a2"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

password_hash = PasswordHash.recommended()


DUMMY_HASH = password_hash.hash("dummypassword")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


def get_user(db_session: Session, username: str) -> HumanUserModel | None:
    stmt = select(HumanUserModel).where(HumanUserModel.username == username)
    user = db_session.scalars(stmt).one_or_none()
    if user is not None:
        return user


def authenticate_user(
    db_session: Session, username: str, password: str
) -> HumanUserModel | Literal[False]:
    user = get_user(db_session, username)
    if not user:
        password_hash.verify(password, DUMMY_HASH)
        return False
    if not password_hash.verify(password, user.hashed_password):
        return False
    return user


def create_access_token(data: dict, expires_delta: timedelta | None = None) -> str:
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(UTC) + expires_delta
    else:
        expire = datetime.now(UTC) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def get_current_user(
    db_session: Annotated[Session, Depends(get_db_session)],
    token: Annotated[str, Depends(oauth2_scheme)],
) -> HumanUserModel:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if username is None:
            raise credentials_exception
    except jwt.InvalidTokenError:
        raise credentials_exception
    user = get_user(db_session, username=username)
    if user is None:
        raise credentials_exception
    return user


class Token(BaseModel):
    access_token: str
    token_type: str


def generate_token(username: str) -> Token:
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": username}, expires_delta=access_token_expires
    )
    return Token(access_token=access_token, token_type="bearer")


class BaseHeaders:
    def __init__(
        self,
        agent_id: Annotated[int | None, Header()] = None,
        session_id: Annotated[int | None, Header()] = None,
        parent_session_id: Annotated[int | None, Header()] = None,
    ):
        self.user_id: int | None = None
        self.agent_id = agent_id
        self.session_id = session_id
        self.parent_session_id = parent_session_id


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
    call_id: str
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


def append_output(
    response_output: list[ResponseOutputItem],
    update_func: Callable[[ResponseInputItemParam | HumanInputFromUser], None],
) -> None:
    for item in response_output:
        if item.type == "reasoning":
            item_param = ResponseReasoningItemParam(item.model_dump())
        elif item.type == "message":
            item_param = ResponseOutputMessageParam(item.model_dump())
        elif item.type == "function_call":
            item_param = ResponseFunctionToolCallParam(item.model_dump())
        update_func(item_param)


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
    update_func: Callable[[ResponseInputItemParam | HumanInputFromUser], None],
    pending_call_ids: dict[
        str, tuple[ResponseFunctionToolCall, ResponseFunctionToolCallOutputItem]
    ],
) -> list[ServerSentEvent]:
    res = await process_tool_call(tool_call=tool_call, tool=tools_dict[tool_call.name])
    call_output = res.output
    assert isinstance(call_output, list)
    res_list: list[ServerSentEvent] = []
    for item in call_output:
        if item.type == "input_file":
            res_list.append(ServerSentEvent(data=item))
    if res.status == "in_progress":
        pending_call_ids[res.call_id] = (
            tool_call,
            res,
        )
        res_list.append(
            ServerSentEvent(
                data=HumanInputRequired(
                    type="human_input_required",
                    tool_call_param=tool_call,
                    tool_call_output=res,
                )
            )
        )
    update_func(FunctionCallOutput(res.model_dump()))
    return res_list


app = FastAPI()


async def run_model_loop(
    input_list: ResponseInputParam,
    pending_call_ids: dict[
        str, tuple[ResponseFunctionToolCall, ResponseFunctionToolCallOutputItem]
    ],
    update_func: Callable[[ResponseInputItemParam | HumanInputFromUser], None],
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
        append_output(response_output=response.output, update_func=update_func)
        for item in response.output:
            if item.type == "function_call":
                tool_calls.append(item)
            elif item.type == "message":
                yield ServerSentEvent(data=item)
        if not tool_calls:
            break
        tasks = [
            asyncio.create_task(
                run_tool_call(
                    tool_call=tool_call,
                    tools_dict=tools_dict,
                    update_func=update_func,
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


class UserCreate(BaseModel):
    username: str
    password: str
    fullname: str
    email: str


class UserResponse(BaseModel):
    username: str
    user_id: int
    token: Token


@app.post("/signup", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def signup(
    user_details: UserCreate, db_session: Annotated[Session, Depends(get_db_session)]
):
    # Base.metadata.create_all(db_engine)
    stmt = select(HumanUserModel).where(
        HumanUserModel.username == user_details.username
    )
    prev_user = db_session.scalars(stmt).first()
    if prev_user:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail="User already exists"
        )
    new_user = HumanUserModel(
        username=user_details.username,
        hashed_password=password_hash.hash(user_details.password),
        fullname=user_details.fullname,
        email=user_details.email,
        user=UserModel(),
    )
    db_session.add(new_user)
    db_session.commit()
    return UserResponse(
        username=new_user.username,
        user_id=new_user.user_id,
        token=generate_token(new_user.username),
    )


@app.post("/token")
async def login_for_access_token(
    db_session: Annotated[Session, Depends(get_db_session)],
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
) -> Token:
    user = authenticate_user(db_session, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return generate_token(user.username)


allowed_items = {"reasoning", "message", "function_call", "function_call_output"}


def messages_to_input_list(
    message_list: Sequence[MessageModel],
) -> tuple[
    ResponseInputParam,
    dict[str, tuple[ResponseFunctionToolCall, ResponseFunctionToolCallOutputItem]],
]:
    input_list = []
    pending_call_ids: dict[str, list] = {}
    for message in message_list:
        message_dict = json.loads(message.content)
        if message_dict.get("type") in allowed_items:
            if message_dict.get("type") == "function_call":
                pending_call_ids[message_dict.get("call_id")] = [
                    ResponseFunctionToolCall(**message_dict),
                    None,
                ]
            elif message_dict.get("type") == "function_call_output":
                if message_dict.get("status") == "completed":
                    # pending_call_ids.remove(message_dict.get("call_id"))
                    pending_call_ids.pop(message_dict.get("call_id"))
                elif message_dict.get("status") == "in_progress":
                    pending_call_ids[message_dict.get("call_id")][1] = (
                        ResponseFunctionToolCallOutputItem(**message_dict)
                    )
                else:
                    continue

        input_list.append(message_dict)
    return input_list, {k: tuple(v) for k, v in pending_call_ids.items()}


def message_update_input_list_db(
    item: ResponseInputItemParam | HumanInputFromUser,
    input_list: ResponseInputParam,
    base_headers: BaseHeaders,
    db_session: Session,
) -> None:
    if isinstance(item, HumanInputFromUser):
        db_session.add(
            MessageModel(
                session_id=base_headers.session_id,
                parent_session_id=base_headers.parent_session_id,
                content=item.model_dump_json(),
                message_time=datetime.now(UTC),
            )
        )
    else:
        db_session.add(
            MessageModel(
                session_id=base_headers.session_id,
                parent_session_id=base_headers.parent_session_id,
                content=json.dumps(item),
                message_time=datetime.now(UTC),
            )
        )
    db_session.commit()
    if (isinstance(item, dict)) and (item.get("type") in allowed_items):
        if (item.get("type") == "function_call_output") and (
            item.get("status") != "completed"
        ):
            return
        input_list.append(item)


# Need to break up this function and remove print statements
@app.post("/", response_class=EventSourceResponse)
async def run_model(
    model_input: EasyInputMessage | HumanInputFromUser,
    db_session: Annotated[Session, Depends(get_db_session)],
    base_headers: Annotated[BaseHeaders, Depends()],
    user: Annotated[HumanUserModel, Depends(get_current_user)],
) -> AsyncIterable[ServerSentEvent]:
    base_headers.user_id = user.user_id
    if base_headers.session_id is None:
        session = SessionModel(user_id=user.user_id)
        db_session.add(session)
        db_session.commit()
        base_headers.session_id = session.session_id
        yield ServerSentEvent(
            data={"type": "session_init", "session_id": session.session_id}
        )
        message_list = []
    else:
        stmt = (
            select(MessageModel)
            .where(MessageModel.session_id == base_headers.session_id)
            .order_by(MessageModel.message_time)
        )
        message_list = db_session.scalars(stmt).fetchall()
    input_list, pending_call_ids = messages_to_input_list(message_list)
    input_list_db_update_func = partial(
        message_update_input_list_db,
        input_list=input_list,
        base_headers=base_headers,
        db_session=db_session,
    )
    if isinstance(model_input, EasyInputMessage):
        if pending_call_ids:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="We should not be getting input while tool call human input is pending",
            )
        input_list_db_update_func(model_input.model_dump())
        # Pending_call_ids is modified in place
        async for sse_event in run_model_loop(
            input_list=input_list,
            pending_call_ids=pending_call_ids,
            update_func=input_list_db_update_func,
        ):
            yield sse_event
    elif isinstance(model_input, HumanInputFromUser):
        if model_input.call_id not in pending_call_ids:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="This function call has already been processed",
            )
        tool_call_param, tool_call_output = pending_call_ids[model_input.call_id]
        arguments_json: dict[str, Any] = json.loads(tool_call_param.arguments)
        assert isinstance(tool_call_output.output[1], ResponseInputText)
        model_def = tool_call_output.output[1].text
        human_input_model = make_pydantic_model_from_def(model_def)
        human_input = human_input_model(**model_input.human_input)
        arguments_json.update(human_input.model_dump())
        arguments_json["human_input"] = True
        tool_call_copy = tool_call_param.model_copy(
            update={"arguments": json.dumps(arguments_json)}
        )
        try:
            pending_call_ids.pop(model_input.call_id)
            for sse_event in await run_tool_call(
                tool_call=tool_call_copy,
                tools_dict=tools_dict,
                update_func=input_list_db_update_func,
                pending_call_ids=pending_call_ids,
            ):
                yield sse_event
        except Exception:
            pending_call_ids[model_input.call_id] = (tool_call_param, tool_call_output)
        if not pending_call_ids:
            async for sse_event in run_model_loop(
                input_list=input_list,
                pending_call_ids=pending_call_ids,
                update_func=input_list_db_update_func,
            ):
                yield sse_event
