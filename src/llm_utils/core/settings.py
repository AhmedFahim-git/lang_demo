from typing import ClassVar

from openai import AsyncOpenAI
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")
    agent_base_url: str
    a2a_base_url: str
    a2a_api_key: str
    auth_token_secret_key: str
    openai_client: ClassVar[AsyncOpenAI] = AsyncOpenAI(
        api_key="None", base_url="http://localhost:8080/v1"
    )


settings = Settings()
