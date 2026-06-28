import asyncio

import httpx
from a2a.client import A2ACardResolver, ClientConfig, create_client
from a2a.helpers import new_text_message
from a2a.types import Role, SendMessageRequest


async def main():
    async with httpx.AsyncClient() as httpx_client:
        resolver = A2ACardResolver(
            httpx_client=httpx_client, base_url="http://127.0.0.1:8000"
        )
        public_card = await resolver.get_agent_card()

    print(public_card)
    config = ClientConfig(streaming=True)
    client = await create_client(agent=public_card, client_config=config)

    message = new_text_message(text="1", role=Role.ROLE_USER)
    request = SendMessageRequest(message=message)
    async for chunk in client.send_message(request=request):
        print(type(chunk))
        if chunk.message.parts:
            print(type(chunk.message))
            print("na uh")
        print("wassup")
        print(chunk)
        print("sup")


if __name__ == "__main__":
    asyncio.run(main())
