from sqlalchemy import select
from sqlalchemy.orm import Session

from llm_utils.db.schema import SessionModel, UserModel
from llm_utils.models.session_models import SessionCreate


class SessionService:
    def __init__(self, db_session: Session):
        self._db_session = db_session

    def make_chat_session(self, user_id: int) -> SessionCreate:
        session = SessionModel(user_id=user_id)
        self._db_session.add(session)
        self._db_session.commit()
        return SessionCreate(session_id=session.session_id)

    def get_user_from_session_id(self, session_id: int) -> UserModel | None:
        stmt = select(SessionModel).where(SessionModel.session_id == session_id)
        session = self._db_session.scalars(stmt).one_or_none()
        user = session.user if session else None
        return user

    def validate_user_session(self, user_id: int, session_id: int) -> bool:
        stmt = select(SessionModel).where(
            SessionModel.user_id == user_id, SessionModel.session_id == session_id
        )
        session = self._db_session.scalars(stmt).one_or_none()
        return session is not None
