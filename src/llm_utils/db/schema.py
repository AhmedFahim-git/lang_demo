from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
)


class Base(DeclarativeBase):
    pass


class HumanUserModel(Base):
    __tablename__ = "human_users"

    human_user_id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    username: Mapped[str] = mapped_column(
        String(length=50), unique=True, nullable=False, index=True
    )
    hashed_password: Mapped[str] = mapped_column(String(length=200), nullable=False)
    fullname: Mapped[str] = mapped_column(String(length=50), nullable=False)
    email: Mapped[str] = mapped_column(String(length=50), unique=True, nullable=False)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.user_id"), nullable=False, unique=True
    )

    user: Mapped["UserModel"] = relationship(back_populates="human_user")


class AgentUserModel(Base):
    __tablename__ = "agent_users"

    agent_user_id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    agent_name: Mapped[str] = mapped_column(String(length=50), nullable=False)
    agent_description: Mapped[str] = mapped_column(Text, nullable=False)
    system_prompt: Mapped[str] = mapped_column(Text, nullable=False)
    tools_list: Mapped[str] = mapped_column(
        Text, default="[]", nullable=False
    )  # Comma separated list of tool names (can be more sophisticated later on)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.user_id"), nullable=False, unique=True
    )

    user: Mapped["UserModel"] = relationship(back_populates="agent_user")


class UserModel(Base):
    __tablename__ = "users"

    user_id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    is_human: Mapped[bool] = mapped_column(Boolean, default=True)

    human_user: Mapped["HumanUserModel"] = relationship(
        back_populates="user", foreign_keys=[HumanUserModel.user_id]
    )
    agent_user: Mapped["AgentUserModel"] = relationship(
        back_populates="user", foreign_keys=[AgentUserModel.user_id]
    )
    sessions: Mapped[list["SessionModel"]] = relationship(back_populates="user")


class MessageModel(Base):
    __tablename__ = "messages"

    message_id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    session_id: Mapped[int] = mapped_column(
        ForeignKey("sessions.session_id"), nullable=False, index=True
    )
    parent_session_id: Mapped[int] = mapped_column(
        ForeignKey("sessions.session_id"), nullable=True
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    message_time: Mapped[datetime] = mapped_column(DateTime(timezone=True))

    session: Mapped["SessionModel"] = relationship(
        back_populates="messages", foreign_keys=[session_id]
    )
    parent_session: Mapped["SessionModel"] = relationship(
        back_populates="messages_as_parent", foreign_keys=[parent_session_id]
    )

    def __repr__(self):
        return f"Message_id: {self.message_id}, content: {self.content}, session_id: {self.session_id}, message_time: {self.message_time}"


class SessionModel(Base):
    __tablename__ = "sessions"

    session_id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.user_id"), nullable=False, index=True
    )

    user: Mapped["UserModel"] = relationship(back_populates="sessions")
    messages: Mapped[list["MessageModel"]] = relationship(
        back_populates="session", foreign_keys=[MessageModel.session_id]
    )
    messages_as_parent: Mapped[list["MessageModel"]] = relationship(
        back_populates="parent_session", foreign_keys=[MessageModel.parent_session_id]
    )
