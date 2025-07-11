from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    openai_apikey: str

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
