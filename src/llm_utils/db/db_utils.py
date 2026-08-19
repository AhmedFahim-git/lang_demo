import json
from collections.abc import Iterator

from sqlalchemy import (
    StaticPool,
    create_engine,
    select,
)
from sqlalchemy.orm import (
    Session,
    sessionmaker,
)

from .schema import AgentUserModel, Base, UserModel

SQLITE_DB_NAME = "sqlite_db.db"

# if os.path.exists(SQLITE_DB_NAME):
#     os.remove(SQLITE_DB_NAME)

DB_URL = f"sqlite:///{SQLITE_DB_NAME}"

# Apparently this particular configuration is needed for sql in memory to work (look it up)
db_engine = create_engine(
    url=DB_URL, connect_args={"check_same_thread": False}, poolclass=StaticPool
)

Base.metadata.create_all(db_engine)

SessionLocal = sessionmaker(bind=db_engine)


def get_db_session() -> Iterator[Session]:
    with SessionLocal() as db_session:
        yield db_session


def make_agents():
    session = SessionLocal()
    # TODO: Add other agents
    agents = [
        AgentUserModel(
            agent_user_id=1,
            agent_name="base_agent",
            agent_description="Base Agent",
            system_prompt="You are a helpful agent. If user asks to get whether a number is prime you should use the prime_agent tool/subagent and return its result.",
            tools_list=json.dumps(
                ["use_browser", "get_weather", "get_time", "prime_agent"]
            ),
            user=UserModel(),
        ),
        AgentUserModel(
            agent_name="prime_agent",
            agent_description="Classify a given input number as prime or not prime.",
            system_prompt="You are a helpful agent. Your main task is to classify a given number as prime or not prime.",
            user=UserModel(),
        ),
    ]
    session.add_all(agents)
    session.commit()


def get_agents(db_session: Session) -> list[AgentUserModel]:
    stmt = select(AgentUserModel).where(AgentUserModel.agent_user_id != 1)
    agents = list(db_session.scalars(stmt).fetchall())
    return agents
