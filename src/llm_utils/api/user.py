from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from llm_utils.auth.auth import generate_token
from llm_utils.db.db_utils import get_db_session
from llm_utils.models.user_models import UserCreate, UserResponse
from llm_utils.services.user_service import HumanUserService

router = APIRouter()


def make_human_user_service(
    db_session: Annotated[Session, Depends(get_db_session)],
) -> HumanUserService:
    return HumanUserService(db_session=db_session)


@router.post(
    "/signup", response_model=UserResponse, status_code=status.HTTP_201_CREATED
)
async def signup_user(
    user_details: UserCreate,
    user_service: Annotated[HumanUserService, Depends(make_human_user_service)],
) -> UserResponse:
    user = user_service.create_user(user_details)
    if user:
        return UserResponse(
            username=user.username,
            user_id=user.user_id,
            token=generate_token(user.username),
        )
    else:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail="User already exists"
        )
