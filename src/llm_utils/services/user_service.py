import jwt
from sqlalchemy import select
from sqlalchemy.orm import Session

from llm_utils.auth.auth import (
    DUMMY_HASH,
    decode_token,
    generate_token,
    hash_password,
    verify_password,
)
from llm_utils.db.schema import HumanUserModel, UserModel
from llm_utils.models.auth_models import Token
from llm_utils.models.user_models import UserCreate


class UserService:
    def __init__(self, db_session: Session):
        self._db_session = db_session


class HumanUserService(UserService):
    def create_user(self, user_details: UserCreate) -> HumanUserModel | None:
        stmt = select(HumanUserModel).where(
            HumanUserModel.username == user_details.username
        )
        prev_user = self._db_session.scalars(stmt).first()
        if prev_user:
            return None
        new_user = HumanUserModel(
            username=user_details.username,
            hashed_password=hash_password(user_details.password),
            fullname=user_details.fullname,
            email=user_details.email,
            user=UserModel(),
        )
        self._db_session.add(new_user)
        self._db_session.commit()
        return new_user

    def _get_user(self, username: str) -> HumanUserModel | None:
        stmt = select(HumanUserModel).where(HumanUserModel.username == username)
        user = self._db_session.scalars(stmt).one_or_none()
        return user

    def authenticate_user(self, username: str, password: str) -> HumanUserModel | None:
        user = self._get_user(username)
        if not user:
            verify_password(password, DUMMY_HASH)
            return None
        if not verify_password(password, user.hashed_password):
            return None
        return user

    def get_current_user(self, token: str) -> HumanUserModel | None:
        try:
            payload = decode_token(token)
            username = payload.get("sub")
            if username is None:
                return None
        except jwt.InvalidTokenError:
            return None
        assert isinstance(username, str)
        user = self._get_user(username=username)
        return user

    def generate_user_token(self, username: str, password: str) -> Token | None:
        user = self.authenticate_user(username=username, password=password)
        if user:
            return generate_token(username=username)
        return None

    def get_human_user_from_user_id(self, user_id: int) -> HumanUserModel | None:
        stmt = select(HumanUserModel).where(HumanUserModel.user_id == user_id)
        user = self._db_session.scalars(stmt).one_or_none()
        return user
