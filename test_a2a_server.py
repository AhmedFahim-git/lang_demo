import time

import httpx
import uvicorn
from a2a.helpers import (
    get_message_text,
    new_task_from_user_message,
    new_text_message,
    new_text_part,
)
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import (
    add_a2a_routes_to_fastapi,
    create_agent_card_routes,
    create_jsonrpc_routes,
)
from a2a.server.tasks import (
    BasePushNotificationSender,
    InMemoryPushNotificationConfigStore,
    InMemoryTaskStore,
    TaskUpdater,
)
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    AgentProvider,
    AgentSkill,
    Message,
    Part,
    Role,
    TaskState,
)
from fastapi import FastAPI
from google.protobuf.json_format import MessageToDict, ParseDict
from google.protobuf.struct_pb2 import Value


class SampleAgent:
    async def invoke(self, message: str) -> str:
        if message == "1":
            return "This is quick response"
        elif message == "2":
            time.sleep(5)
            return "This is slow response"
        else:
            return "This is default response"


class SampleAgentExecutor(AgentExecutor):
    def __init__(self):
        self.agent = SampleAgent()

    async def execute(self, context: RequestContext, event_queue: EventQueue):
        assert context.message
        print(context)
        print(context.current_task)
        print(event_queue)
        # await event_queue.enqueue_event(event=new_text_message("Yo what up"))

        # task = context.current_task or new_task_from_user_message(context.message)
        if context.current_task:
            task = context.current_task
        else:
            task = new_task_from_user_message(context.message)
            await event_queue.enqueue_event(task)

        task_updater = TaskUpdater(
            event_queue=event_queue, task_id=task.id, context_id=task.context_id
        )
        if task.status.state == TaskState.TASK_STATE_INPUT_REQUIRED:
            last_part = MessageToDict(context.message.parts[-1])
            new_part = Part(data=ParseDict(last_part["data"], Value()))
            update_message = Message(
                role=Role.ROLE_AGENT,
                context_id=task.context_id,
                task_id=task.id,
                parts=[
                    new_part,
                ],
            )
            # await task_updater.update_status(state=TaskState.TASK_STATE_WORKING, message=update_message)
            await task_updater.start_work(message=update_message)
            await task_updater.add_artifact(
                parts=[new_text_part("Yo this is my message")]
            )
            await task_updater.complete(
                new_text_message(text="Task Completed Successfully")
            )

        else:
            await task_updater.update_status(
                state=TaskState.TASK_STATE_WORKING,
                message=new_text_message("Start working on task", role=Role.ROLE_AGENT),
            )

            query = get_message_text(context.message)
            if query:
                result = await self.agent.invoke(query)
            else:
                result = "No text input provided"

            await task_updater.add_artifact(
                parts=[new_text_part(text=result, media_type="text/plain")]
            )
            print("result:", result)
            await task_updater.requires_input(
                message=new_text_message(
                    text="Give me some input",
                    role=Role.ROLE_AGENT,
                    context_id=task.context_id,
                    task_id=task.id,
                )
            )

        # await task_updater.update_status(
        #     state=TaskState.TASK_STATE_COMPLETED,
        #     message=new_text_message(text="Task Completed Successfully"),
        # )
        # await event_queue.enqueue_event(
        #     new_task_from_user_message(
        #         new_text_message("My Message", role=Role.ROLE_USER)
        #     )
        # )

    async def cancel(self, context: RequestContext, event_queue: EventQueue):
        task = context.current_task
        assert task
        task_updater = TaskUpdater(
            event_queue=event_queue, task_id=task.id, context_id=task.context_id
        )
        # await task_updater.update_status(
        #     state=TaskState.TASK_STATE_CANCELED,
        #     message=new_text_message(text="Task Cancelled Successfully"),
        # )
        await task_updater.cancel()


if __name__ == "__main__":
    app = FastAPI()

    skill = AgentSkill(
        id="sample_a2a_skill",
        name="Sample A2A Skill",
        description="This is my Sample A2A skill",
        examples=["1", "2", "3"],
        input_modes=["text/plain"],
        output_modes=["text/plain"],
    )

    agent_card = AgentCard(
        name="My Sample A2A agent",
        description="This is my sample A2A agent",
        supported_interfaces=[
            AgentInterface(url="http://127.0.0.1:8000", protocol_binding="JSONRPC")
        ],
        provider=AgentProvider(url="https://github.com/", organization="Github"),
        capabilities=AgentCapabilities(streaming=True, push_notifications=False),
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        skills=[skill],
    )
    httpx_client = httpx.AsyncClient()
    push_config_store = InMemoryPushNotificationConfigStore()
    push_sender = BasePushNotificationSender(
        httpx_client=httpx_client, config_store=push_config_store
    )

    request_handler = DefaultRequestHandler(
        agent_executor=SampleAgentExecutor(),
        task_store=InMemoryTaskStore(),
        agent_card=agent_card,
        push_config_store=push_config_store,
        push_sender=push_sender,
    )

    add_a2a_routes_to_fastapi(
        app=app,
        agent_card_routes=create_agent_card_routes(agent_card=agent_card),
        jsonrpc_routes=create_jsonrpc_routes(
            request_handler=request_handler, rpc_url="/"
        ),
    )

    uvicorn.run(app)
