from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from llm_utils.auth.auth import generate_token
from llm_utils.db.db_utils import get_db_session
from llm_utils.db.schema import HumanUserModel
from llm_utils.models.auth_models import Token
from llm_utils.services.user_service import HumanUserService

router = APIRouter()


def make_human_user_service(
    db_session: Annotated[Session, Depends(get_db_session)],
) -> HumanUserService:
    return HumanUserService(db_session=db_session)


def authenticate_user(
    human_user_service: Annotated[HumanUserService, Depends(make_human_user_service)],
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
) -> HumanUserModel | None:
    return human_user_service.authenticate_user(form_data.username, form_data.password)


@router.post("/token")
async def login_for_access_token(
    user: Annotated[HumanUserModel | None, Depends(authenticate_user)],
) -> Token:
    if user:
        return generate_token(user.username)
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
