import asyncio

import httpx
from a2a.client import A2ACardResolver, ClientConfig, create_client
from a2a.helpers import new_text_message
from a2a.types import (
    Role,
    SendMessageRequest,
    TaskPushNotificationConfig,
)


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
        print(type(chunk))
        if chunk.message.parts:
            print(type(chunk.message))
            print("na uh")
        print("wassup")
        print(chunk)
        print("sup")


async def main():
    res = await a2a_agent_call(base_url="http://127.0.0.1:8000", str_message="1")


if __name__ == "__main__":
    asyncio.run(main())
