from fastapi import FastAPI

from llm_utils.api import auth, chat, user

app = FastAPI()

app.include_router(user.router, prefix="/user")
app.include_router(auth.router, prefix="/auth")
app.include_router(chat.router, prefix="/chat")
