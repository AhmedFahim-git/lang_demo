from pydantic import BaseModel

from .auth_models import Token


class UserCreate(BaseModel):
    username: str
    password: str
    fullname: str
    email: str


class UserResponse(BaseModel):
    username: str
    user_id: int
    token: Token
