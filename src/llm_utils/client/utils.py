import asyncio
import json
from typing import Any

import httpx

from llm_utils.core.settings import settings


async def get_input(message: str) -> str:
    return await asyncio.to_thread(input, message)


async def display_message(message: Any) -> None:
    print(message)


def parse_sse(text: str) -> dict[str, Any]:
    items: dict[str, Any] = {}
    for line in text.strip().splitlines():
        k, v = line.split(sep=":", maxsplit=1)
        items[k] = v.strip()
    if "data" in items:
        items["data"] = json.loads(items["data"])
    return items


async def process_human_input_request(
    input_request: dict[str, Any], token: str, session_id: int | None = None
):
    tool_call_param, tool_call_output = (
        input_request["tool_call_param"],
        input_request["tool_call_output"],
    )
    await display_message(tool_call_param)
    await display_message(tool_call_output)
    assert isinstance(tool_call_output["output"], list)
    assert tool_call_output["output"][1].get("type") == "input_text"
    message_def_str: str = tool_call_output["output"][1].get("text")
    assert message_def_str
    message_def = json.loads(message_def_str)

    assert (
        isinstance(message_def, dict)
        and "fields" in message_def
        and all("field_params" in item for item in message_def["fields"])
        and all("name" in item for item in message_def["fields"])
    )
    required_fields: list[str] = [
        field["name"]
        for field in message_def["fields"]
        if "default" not in field["field_params"]
    ]
    # TODO: Replace while with for loop to limit max number of retries. Keep max retries in a Constant
    while True:
        # try:
        human_input = await get_input(
            "Provide Requested information in json format: "
        )  # It should be {"feedback":"some feedback"}
        # human_input = '{"feedback": "some feedback"}'
        human_json: dict[str, Any] = json.loads(human_input)
        assert all(field in human_json for field in required_fields)
        await send_message(
            {
                "type": "human_input_from_user",
                "call_id": tool_call_param.get("call_id"),
                "human_input": human_json,
            },
            token=token,
            session_id=session_id,
        )
        break
    # except Exception as e:
    #     await display_message(str(e))


async def send_message(
    user_input: dict[str, Any], token: str, session_id: int | None = None
):
    header = {"Authorization": f"Bearer {token}"}
    timeout = httpx.Timeout(10, read=None)
    if session_id is None:
        async with httpx.AsyncClient() as client:
            res = await client.get(
                "http://localhost:8000/chat", headers=header, timeout=timeout
            )
        res.raise_for_status()
        session_id = res.json().get("session_id")
    try:
        async with (
            asyncio.TaskGroup() as tg,
            httpx.AsyncClient() as client,
            client.stream(
                method="POST",
                url=f"{settings.agent_base_url}/chat/{session_id}",
                json=user_input,
                headers=header,
                timeout=timeout,
            ) as stream_res,
        ):
            stream_res.raise_for_status()
            async for text in stream_res.aiter_text():
                items = parse_sse(text)
                if "data" not in items:
                    continue
                data: dict[str, Any] = items["data"]
                if data["type"] == "message":
                    assert isinstance(data["content"], list)
                    for item in data["content"]:
                        if item["type"] == "refusal":
                            await display_message(item["refusal"])
                        if item["type"] == "output_text":
                            await display_message(item["text"])
                elif data["type"] == "input_file":
                    await display_message(data["file_url"])
                elif data["type"] == "human_input_required":
                    tg.create_task(
                        process_human_input_request(
                            data, token=token, session_id=session_id
                        )
                    )
    except ExceptionGroup as e:
        print(e.exceptions)
