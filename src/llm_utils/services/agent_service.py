import asyncio
import json
import uuid
from collections.abc import AsyncIterable, Awaitable, Callable, Sequence
from datetime import UTC, datetime
from inspect import signature
from typing import Any

from fastapi import HTTPException, status
from openai import AsyncOpenAI
from openai.types.responses import (
    EasyInputMessage,
    ResponseFunctionToolCall,
    ResponseFunctionToolCallOutputItem,
    ResponseFunctionToolCallParam,
    ResponseInputFile,
    ResponseInputImage,
    ResponseInputItemParam,
    ResponseInputParam,
    ResponseInputText,
    ResponseOutputItem,
    ResponseOutputMessage,
    ResponseOutputMessageParam,
    ResponseReasoningItemParam,
)
from openai.types.responses.response_input_param import FunctionCallOutput
from sqlalchemy import case, or_, select
from sqlalchemy.orm import Session

from llm_utils.db.schema import AgentUserModel, MessageModel
from llm_utils.models.model_utils import make_pydantic_model_from_def
from llm_utils.models.response_models import HumanInputFromUser, HumanInputRequired
from llm_utils.models.tool_models import ToolOutput
from llm_utils.tools.tools import get_tool_json_list, get_tools_dict

from .session_service import SessionService


class AgentService:
    input_list_allowed_items = frozenset(
        ["reasoning", "message", "function_call", "function_call_output"]
    )

    def __init__(
        self, client: AsyncOpenAI, session_id: int, db_session: Session
    ) -> None:
        self.client = client
        self.session_id = session_id
        self._db_session = db_session
        user = SessionService(db_session).get_user_from_session_id(session_id)
        assert user is not None
        stmt = (
            select(AgentUserModel)
            .where(
                or_(
                    AgentUserModel.user_id == user.user_id,
                    AgentUserModel.agent_user_id == 1,
                )
            )
            .order_by(case((AgentUserModel.user_id == user.user_id, 0), else_=1))
            .limit(1)
        )
        agent = db_session.scalars(stmt).one()
        self.system_prompt = agent.system_prompt  # "You are a helpful agent"
        self.tools: list[str] = json.loads(
            agent.tools_list
        )  # ["use_browser", "get_weather", "get_time"]
        self.tools_dict = get_tools_dict(db_session=db_session)
        self.tools_json_list = get_tool_json_list(
            [self.tools_dict[item] for item in self.tools]
        )

        messages = self._get_prev_messages()
        self.input_list, self.pending_call_ids = self._messages_to_input_list(messages)

    def _get_prev_messages(self) -> list[MessageModel]:
        stmt = (
            select(MessageModel)
            .where(MessageModel.session_id == self.session_id)
            .order_by(MessageModel.message_time)
        )
        return list(self._db_session.scalars(stmt).fetchall())

    @classmethod
    def _messages_to_input_list(
        cls,
        message_list: Sequence[MessageModel],
    ) -> tuple[
        ResponseInputParam,
        dict[str, tuple[ResponseFunctionToolCall, ResponseFunctionToolCallOutputItem]],
    ]:
        input_list = []
        pending_call_ids: dict[str, list] = {}
        for message in message_list:
            message_dict = json.loads(message.content)
            if message_dict.get("type") in cls.input_list_allowed_items:
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

    def _message_update_input_list_db(
        self, item: ResponseInputItemParam | HumanInputFromUser
    ) -> None:
        if isinstance(item, HumanInputFromUser):
            self._db_session.add(
                MessageModel(
                    session_id=self.session_id,
                    content=item.model_dump_json(),
                    message_time=datetime.now(UTC),
                )
            )
        else:
            self._db_session.add(
                MessageModel(
                    session_id=self.session_id,
                    content=json.dumps(item),
                    message_time=datetime.now(UTC),
                )
            )
        self._db_session.commit()
        if (isinstance(item, dict)) and (
            item.get("type") in self.input_list_allowed_items
        ):
            if (item.get("type") == "function_call_output") and (
                item.get("status") != "completed"
            ):
                return
            self.input_list.append(item)

    def _append_output(self, response_output: list[ResponseOutputItem]) -> None:
        for item in response_output:
            if item.type == "reasoning":
                item_param = ResponseReasoningItemParam(item.model_dump())
            elif item.type == "message":
                item_param = ResponseOutputMessageParam(item.model_dump())
            elif item.type == "function_call":
                item_param = ResponseFunctionToolCallParam(item.model_dump())
            else:
                raise Exception(f"Unknown item type: {item.type}")
            self._message_update_input_list_db(item_param)

    @staticmethod
    async def process_tool_call(
        tool_call: ResponseFunctionToolCall,
        tool: Callable[..., Awaitable[ToolOutput]],
    ) -> ResponseFunctionToolCallOutputItem:
        tool_annotation = signature(tool).parameters["args"].annotation
        result = await tool(tool_annotation(**json.loads(tool_call.arguments)))
        assert result.status.value in {"in_progress", "completed", "incomplete"}
        return ResponseFunctionToolCallOutputItem(
            call_id=tool_call.call_id,
            output=result.result,
            type="function_call_output",
            id=uuid.uuid4().hex,
            status=result.status.value,
        )

    async def _run_tool_call(
        self,
        tool_call: ResponseFunctionToolCall,
    ) -> list[
        ResponseInputText | ResponseInputImage | ResponseInputFile | HumanInputRequired
    ]:
        # ) -> list[ServerSentEvent]:
        res = await self.process_tool_call(
            tool_call=tool_call, tool=self.tools_dict[tool_call.name]
        )
        call_output = res.output
        assert isinstance(call_output, list)
        res_list: list[
            ResponseInputText
            | ResponseInputImage
            | ResponseInputFile
            | HumanInputRequired
        ] = []
        for item in call_output:
            if item.type == "input_file":
                res_list.append(item)
        if res.status == "in_progress":
            self.pending_call_ids[res.call_id] = (
                tool_call,
                res,
            )
            res_list.append(
                HumanInputRequired(
                    type="human_input_required",
                    tool_call_param=tool_call,
                    tool_call_output=res,
                )
            )

        self._message_update_input_list_db(FunctionCallOutput(res.model_dump()))
        return res_list

    async def _run_model_loop(
        self,
    ) -> AsyncIterable[
        ResponseOutputMessage
        | ResponseInputText
        | ResponseInputImage
        | ResponseInputFile
        | HumanInputRequired
    ]:
        if self.pending_call_ids:
            # TODO: We should not be dong HTTP exceptions here
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="We should not be create model response while tool call human input is pending",
            )
        tool_calls: list[ResponseFunctionToolCall] = []
        while True:
            response = await self.client.responses.create(
                instructions=self.system_prompt,
                tools=self.tools_json_list,
                input=self.input_list,
            )
            self._append_output(response_output=response.output)
            for item in response.output:
                if item.type == "function_call":
                    tool_calls.append(item)
                elif item.type == "message":
                    yield item
            if not tool_calls:
                break
            tasks = [
                asyncio.create_task(
                    self._run_tool_call(
                        tool_call=tool_call,
                    )
                )
                for tool_call in tool_calls
            ]
            for task in asyncio.as_completed(tasks):
                res = await task
                for item in res:
                    yield item
            if self.pending_call_ids:
                break
            tool_calls = []

    async def run_model(
        self,
        model_input: EasyInputMessage | HumanInputFromUser,
    ) -> AsyncIterable[
        ResponseOutputMessage
        | ResponseInputText
        | ResponseInputImage
        | ResponseInputFile
        | HumanInputRequired
    ]:

        if isinstance(model_input, EasyInputMessage):
            if self.pending_call_ids:
                # TODO: We should not be dong HTTP exceptions here
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="We should not be getting input while tool call human input is pending",
                )
            self._message_update_input_list_db(model_input.model_dump())
            async for item in self._run_model_loop():
                yield item

        elif isinstance(model_input, HumanInputFromUser):
            if model_input.call_id not in self.pending_call_ids:
                # TODO: We should not be dong HTTP exceptions here
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="This function call has already been processed",
                )
            tool_call_param, tool_call_output = self.pending_call_ids[
                model_input.call_id
            ]
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
                self.pending_call_ids.pop(model_input.call_id)
                for model_output_or_input_req in await self._run_tool_call(
                    tool_call=tool_call_copy,
                ):
                    yield model_output_or_input_req
            # TODO: Don't catch exception like this. It is recipe for disaster
            except Exception:
                self.pending_call_ids[model_input.call_id] = (
                    tool_call_param,
                    tool_call_output,
                )
            if not self.pending_call_ids:
                async for model_output_or_input_req in self._run_model_loop():
                    yield model_output_or_input_req
