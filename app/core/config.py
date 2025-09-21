from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    openai_api_base: str = Field(..., validation_alias="OPENAI_API_BASE")
    openai_api_key: str = Field(..., validation_alias="OPENAI_API_KEY")
    openai_model_name: str = Field(..., validation_alias="OPENAI_MODEL_NAME")

    class Config:
        env_file = ".env"


settings = Settings()


if __name__ == "__main__":
    print(settings.model_dump())
