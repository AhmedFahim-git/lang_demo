# Trying out Langchain, WebSearch, Code Execution in Secure Kata Containers, RAG, MCP

# WIP

# Starting llama cpp server

`llama-server -hf Qwen/Qwen3-1.7B-GGUF:Q8_0 --jinja -ngl 99 -fa auto -sm row --temp 0.6 --top-k 20 --top-p 0.95 --min-p 0 --presence-penalty 1.5 -c 40960 -n 32768 --no-context-shift`

# FastAPI dev server

`fastapi dev dir_func.py`

# Curl test command

`curl -F "files=@pyproject.toml" -F "files=@uv.lock" -F "code='print(345)'" -F "pkl_file=@graph_websearch.py" http://127.0.0.1:8000/uploadfile/`

# Gateway API

Install helm charts, minkube tunnel, make gateway api and httproute manifests, update /etc/hosts

The A2A Python repo’s current sender posts a JSON `StreamResponse` to your webhook using `MessageToDict(to_stream_response(event))`, and includes `X-A2A-Notification-Token` when a token is configured. ([GitHub][1]) `StreamResponse` can contain `task`, `message`, `statusUpdate/status_update`, or `artifactUpdate/artifact_update`; push notifications use task/status/artifact events. ([A2A Protocol][2])

Use this FastAPI endpoint:

```python
# notification_receiver.py
import hmac
import logging
import os
from typing import Annotated, Any

from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Request
from google.protobuf.json_format import MessageToDict, ParseDict, ParseError

from a2a.types.a2a_pb2 import StreamResponse

logger = logging.getLogger(__name__)
app = FastAPI()

A2A_PUSH_TOKEN = os.environ.get("A2A_PUSH_TOKEN")


async def handle_a2a_notification(
    payload_type: str,
    task_id: str,
    event: dict[str, Any],
    raw_payload: dict[str, Any],
) -> None:
    """
    Replace this with your DB write, queue publish, websocket broadcast, etc.
    """
    logger.info(
        "Received A2A push notification: type=%s task_id=%s event=%s",
        payload_type,
        task_id,
        event,
    )


def verify_push_token(header_token: str | None) -> None:
    if not A2A_PUSH_TOKEN:
        raise HTTPException(
            status_code=500,
            detail="A2A_PUSH_TOKEN is not configured on this receiver",
        )

    if not header_token or not hmac.compare_digest(header_token, A2A_PUSH_TOKEN):
        raise HTTPException(status_code=401, detail="Invalid A2A notification token")


def parse_a2a_stream_response(payload: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    stream_response = StreamResponse()

    try:
        # Accept lowerCamelCase JSON from MessageToDict, and also proto field names.
        ParseDict(payload, stream_response, ignore_unknown_fields=False)
    except ParseError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid A2A StreamResponse: {exc}") from exc

    payload_type = stream_response.WhichOneof("payload")
    if not payload_type:
        raise HTTPException(status_code=400, detail="Missing StreamResponse payload")

    event_proto = getattr(stream_response, payload_type)
    event_dict = MessageToDict(event_proto, preserving_proto_field_name=True)

    if payload_type == "task":
        task_id = event_proto.id
    elif payload_type in {"status_update", "artifact_update"}:
        task_id = event_proto.task_id
    else:
        # Not expected for push notifications, but keep this explicit.
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported A2A push payload type: {payload_type}",
        )

    if not task_id:
        raise HTTPException(status_code=400, detail="Missing task id in A2A notification")

    return payload_type, task_id, event_dict


@app.post("/a2a/push")
async def receive_a2a_push_notification(
    request: Request,
    background_tasks: BackgroundTasks,
    x_a2a_notification_token: Annotated[
        str | None,
        Header(alias="X-A2A-Notification-Token"),
    ] = None,
) -> dict[str, str]:
    verify_push_token(x_a2a_notification_token)

    payload = await request.json()
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Expected JSON object")

    payload_type, task_id, event = parse_a2a_stream_response(payload)

    # Acknowledge quickly; do heavier work in the background.
    background_tasks.add_task(
        handle_a2a_notification,
        payload_type,
        task_id,
        event,
        payload,
    )

    return {"status": "accepted", "task_id": task_id, "type": payload_type}
```

Run it:

```bash
export A2A_PUSH_TOKEN="use-a-long-random-token"
uvicorn notification_receiver:app --host 0.0.0.0 --port 8000
```

Register this webhook URL in your A2A send configuration as the task push notification config; the spec says `SendMessageConfiguration` has `task_push_notification_config`, and the task id should be empty when sending it with a new `SendMessage` request. ([A2A Protocol][2])

Example URL to give the A2A server:

```text
https://your-domain.example/a2a/push
```

Use the same token value in the A2A `TaskPushNotificationConfig.token`; the sender will place it in `X-A2A-Notification-Token`. ([GitHub][1])

[1]: https://github.com/a2aproject/a2a-python/blob/main/src/a2a/server/tasks/base_push_notification_sender.py "a2a-python/src/a2a/server/tasks/base_push_notification_sender.py at main · a2aproject/a2a-python · GitHub"
[2]: https://a2a-protocol.org/latest/definitions/ "Protocol Definition - A2A Protocol"
