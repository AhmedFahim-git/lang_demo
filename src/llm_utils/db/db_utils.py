from collections.abc import Iterator

from sqlalchemy import (
    StaticPool,
    create_engine,
)
from sqlalchemy.orm import (
    Session,
    sessionmaker,
)

from .schema import Base

DB_URL = "sqlite:///:memory:"

# Apparently this particular configuration is needed for sql in memory to work (look it up)
db_engine = create_engine(
    url=DB_URL, connect_args={"check_same_thread": False}, poolclass=StaticPool
)

Base.metadata.create_all(db_engine)

SessionLocal = sessionmaker(bind=db_engine)


def get_db_session() -> Iterator[Session]:
    with SessionLocal() as db_session:
        yield db_session
