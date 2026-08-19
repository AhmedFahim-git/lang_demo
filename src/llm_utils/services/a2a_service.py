from a2a.helpers import get_message_text, new_text_message
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    AgentSkill,
    Role,
)
from openai.types.responses import (
    EasyInputMessage,
    ResponseInputText,
    ResponseOutputMessage,
)
from sqlalchemy.orm import Session
from starlette.routing import Route

from llm_utils.core.settings import settings

from .agent_service import AgentService
from .session_service import SessionService
from .user_service import AgentUserService


class BaseAgentExecutor(AgentExecutor):
    def __init__(self, db_session: Session, agent_user_id: int):
        self._db_session = db_session
        self.agent_user_id = agent_user_id
        agent = AgentUserService(db_session=db_session).get_user_from_agent_id(
            agent_user_id
        )
        assert agent is not None
        self.agent = agent
        # self.user_id = agent.user_id

    def get_chat_session(self) -> int:
        session_service = SessionService(db_session=self._db_session)
        return session_service.make_chat_session(self.agent.user_id).session_id

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        # TODO: Add mapping from context id to session id
        session_id = self.get_chat_session()
        # TODO: Add security schemes and requirements
        agent_service = AgentService(
            client=settings.openai_client,
            session_id=session_id,
            db_session=self._db_session,
        )
        assert context.message is not None
        # TODO: Add support for task as well, currently only message
        res = agent_service.run_model(
            EasyInputMessage(
                role="user",
                type="message",
                content=[
                    ResponseInputText(
                        type="input_text", text=get_message_text(context.message)
                    )
                ],
            )
        )

        message = new_text_message(text="")
        # TODO: Add support for HumanInputRequired and other return types
        async for item in res:
            if isinstance(item, ResponseOutputMessage):
                for part in item.content:
                    if part.type == "output_text":
                        message = new_text_message(
                            text=part.text,
                            context_id=context.context_id,
                            role=Role.ROLE_AGENT,
                        )
        await event_queue.enqueue_event(event=message)

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        task = context.current_task
        assert task
        task_updater = TaskUpdater(
            event_queue=event_queue, task_id=task.id, context_id=task.context_id
        )
        await task_updater.cancel()


class A2AService:
    def __init__(self, db_session: Session, agent_user_id: int):
        self._db_session = db_session
        self.agent_executor = BaseAgentExecutor(
            db_session=db_session, agent_user_id=agent_user_id
        )
        self.agent_card = self.get_agent_card()
        self.request_handler = DefaultRequestHandler(
            agent_executor=self.agent_executor,
            task_store=InMemoryTaskStore(),
            agent_card=self.agent_card,
        )

    def __del__(self):
        self._db_session.close()

    def get_agent_card(self) -> AgentCard:
        # TODO: Get agent skill id, name etc. from db
        skill = AgentSkill(id="agent_skill", name="Agent skill name")
        interface = AgentInterface(
            url=f"{settings.a2a_base_url}/{self.agent_executor.agent.agent_user_id}",
            protocol_binding="JSONRPC",
        )
        capabilities = AgentCapabilities(streaming=True)
        return AgentCard(
            name=self.agent_executor.agent.agent_name,
            description=self.agent_executor.agent.agent_description,
            supported_interfaces=[interface],
            capabilities=capabilities,
            default_input_modes=["text/plain"],
            default_output_modes=["text/plain"],
            skills=[skill],
        )

    def get_agent_card_routes(self) -> list[Route]:
        return create_agent_card_routes(
            agent_card=self.agent_card,
            card_url=f"/{self.agent_executor.agent.agent_user_id}/.well-known/agent-card.json",
        )

    def get_jsonrpc_routes(self) -> list[Route]:
        return create_jsonrpc_routes(
            request_handler=self.request_handler,
            rpc_url=f"/{self.agent_executor.agent.agent_user_id}",
        )
