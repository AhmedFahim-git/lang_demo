import hmac
from typing import Annotated

import uvicorn
from a2a.types import StreamResponse
from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Request
from google.protobuf.json_format import MessageToDict, ParseDict

app = FastAPI()


async def do_stuff(payload: StreamResponse):
    print(payload)
    payload_type = payload.WhichOneof("payload")
    payload_proto = getattr(payload, payload_type)
    payload_dict = MessageToDict(payload_proto)
    print("payload haha")
    print(payload_dict)


def check_token(token):
    if (not token) or (not hmac.compare_digest(token, "MyToken")):
        raise HTTPException(status_code=401, detail="Not matched")


@app.post("/")
async def get_push_not(
    response: Request,
    background_tasks: BackgroundTasks,
    x_a2a_notification_token: Annotated[str | None, Header()] = None,
):
    print("push this")
    check_token(x_a2a_notification_token)
    payload = await response.json()
    print(payload)
    parsed_payload = StreamResponse()
    ParseDict(payload, parsed_payload)
    background_tasks.add_task(do_stuff, parsed_payload)
    return "received"


uvicorn.run(app, port=8001)
