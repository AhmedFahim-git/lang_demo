from contextlib import asynccontextmanager

import uvicorn
from a2a.server.routes import add_a2a_routes_to_fastapi
from fastapi import FastAPI

from llm_utils.api.a2a import A2A_ROUTES_LIST, A2A_SERVICES
from llm_utils.core.settings import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    for item in A2A_SERVICES:
        await item.request_handler.aclose()


app = FastAPI(lifespan=lifespan)

for route in A2A_ROUTES_LIST:
    add_a2a_routes_to_fastapi(
        app=app,
        agent_card_routes=route.agent_card_routes,
        jsonrpc_routes=route.json_rpc_routes,
    )

uvicorn.run(app, host="0.0.0.0", port=int(settings.a2a_base_url.split(":")[-1]))
