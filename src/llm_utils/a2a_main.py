import secrets
from contextlib import asynccontextmanager

import uvicorn
from a2a.server.routes import add_a2a_routes_to_fastapi
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from starlette.authentication import AuthCredentials, SimpleUser

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


@app.middleware("http")
async def authentication_middleware(request: Request, call_next):
    if request.url.path.endswith("agent-card.json"):
        return await call_next(request)

    api_key = request.headers.get("x-api-key")
    if api_key and secrets.compare_digest(api_key, settings.a2a_api_key):
        request.scope["user"] = SimpleUser("a2a_user")
        request.scope["auth"] = AuthCredentials(["authenticated", "a2a:invoke"])
        return await call_next(request)
    else:
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={"detail": "API KEY is missing"},
        )


uvicorn.run(app, host="0.0.0.0", port=int(settings.a2a_base_url.split(":")[-1]))
