from collections.abc import AsyncIterable
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.sse import EventSourceResponse, ServerSentEvent
from openai.types.responses import EasyInputMessage
from sqlalchemy.orm import Session

from llm_utils.auth.auth import oauth2_scheme
from llm_utils.core.settings import settings
from llm_utils.db.db_utils import get_db_session
from llm_utils.db.schema import UserModel
from llm_utils.models.response_models import HumanInputFromUser
from llm_utils.models.session_models import SessionCreate
from llm_utils.services.agent_service import AgentService
from llm_utils.services.session_service import SessionService
from llm_utils.services.user_service import HumanUserService

router = APIRouter()


# def get_client() -> AsyncOpenAI:
#     return AsyncOpenAI(api_key="None", base_url="http://localhost:8080/v1")


def make_session(
    db_session: Annotated[Session, Depends(get_db_session)],
) -> SessionService:
    return SessionService(db_session=db_session)


def make_human_user_service(
    db_session: Annotated[Session, Depends(get_db_session)],
) -> HumanUserService:
    return HumanUserService(db_session=db_session)


def get_current_user(
    user_service: Annotated[HumanUserService, Depends(make_human_user_service)],
    token: Annotated[str, Depends(oauth2_scheme)],
) -> UserModel | None:
    human_user = user_service.get_current_user(token)
    return human_user.user if human_user else None


@router.get("", response_model=SessionCreate)
async def make_chat_session(
    user: Annotated[UserModel | None, Depends(get_current_user)],
    session_service: Annotated[SessionService, Depends(make_session)],
) -> SessionCreate:
    if user:
        return session_service.make_chat_session(user.user_id)
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


@router.post("/{session_id}", response_class=EventSourceResponse)
async def run_model(
    session_id: int,
    model_input: EasyInputMessage | HumanInputFromUser,
    db_session: Annotated[Session, Depends(get_db_session)],
    user: Annotated[UserModel | None, Depends(get_current_user)],
) -> AsyncIterable[ServerSentEvent]:
    assert user
    assert SessionService(db_session=db_session).validate_user_session(
        user.user_id, session_id
    )
    agent_service = AgentService(
        db_session=db_session, client=settings.openai_client, session_id=session_id
    )
    async for response in agent_service.run_model(model_input):
        yield ServerSentEvent(data=response, retry=5)
