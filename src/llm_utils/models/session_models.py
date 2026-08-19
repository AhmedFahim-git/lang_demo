from pydantic import BaseModel


class SessionCreate(BaseModel):
    type: str = "session_init"
    session_id: int
