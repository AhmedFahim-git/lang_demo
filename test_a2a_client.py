import asyncio
import uuid

import httpx
from a2a.client import A2ACardResolver, ClientConfig, create_client
from a2a.helpers import new_text_message
from a2a.types import (
    Message,
    Part,
    Role,
    SendMessageRequest,
    TaskPushNotificationConfig,
    TaskState,
)
from google.protobuf.json_format import ParseDict
from google.protobuf.struct_pb2 import Value


async def a2a_agent_call(base_url: str, str_message: str):
    async with httpx.AsyncClient() as httpx_client:
        resolver = A2ACardResolver(httpx_client=httpx_client, base_url=base_url)
        public_card = await resolver.get_agent_card()

    print(public_card)
    push_config = TaskPushNotificationConfig(
        url="http://127.0.0.1:8001", token="MyToken"
    )
    config = ClientConfig(streaming=True, push_notification_config=push_config)
    client = await create_client(agent=public_card, client_config=config)

    message = new_text_message(text=str_message, role=Role.ROLE_USER)
    request = SendMessageRequest(message=message)
    async for chunk in client.send_message(request=request):
        chunk.HasField("task")
        print(chunk.task.status)
        print(type(chunk))
        if chunk.message.parts:
            print(type(chunk.message))
            print("na uh")
        print("wassup")
        print(chunk)
        print("sup")

    # Text part
    # text_part = Part(text="What's the weather in Warsaw?")
    # text_part_2 = Part(text="What's the weather?")
    #
    # # Data part — use ParseDict to convert a Python dict to a protobuf Value
    if (
        chunk.HasField("status_update")
        and chunk.status_update.status.state == TaskState.TASK_STATE_INPUT_REQUIRED
    ):
        data_part = Part(
            data=ParseDict({"city": "Warsaw", "temperature_c": 18}, Value()),
        )
        message = Message(
            message_id=uuid.uuid4().hex,
            context_id=chunk.status_update.context_id,
            task_id=chunk.status_update.task_id,
            role=Role.ROLE_USER,
            parts=[data_part],
        )
        request = SendMessageRequest(message=message)
        async for chunk in client.send_message(request=request):
            print(chunk)

    #
    # message = Message(
    #     role=Role.ROLE_USER,
    #     parts=[text_part, data_part],
    # )
    # print(message)
    # print(get_message_text(message))
    # data_p = message.parts.pop()
    # print(data_p)
    # print(MessageToDict(message))
    # print(MessageToDict(message.parts[-1]))


async def main():
    res = await a2a_agent_call(base_url="http://127.0.0.1:8000", str_message="1")


if __name__ == "__main__":
    asyncio.run(main())
