"""
LLM client factory for creating configured language model instances.
"""

from langchain_openai import ChatOpenAI
from core.config import settings


class LLMClient:
    """Factory class for creating configured LLM clients."""

    @staticmethod
    def create_llm() -> ChatOpenAI:
        """
        Create and configure a ChatOpenAI instance.

        Returns:
            ChatOpenAI: Configured LLM client instance.

        Raises:
            ValueError: If required settings are not configured.
        """
        if not all(
            [
                settings.openai_api_base,
                settings.openai_api_key,
                settings.openai_model_name,
            ]
        ):
            raise ValueError(
                "Missing required LLM configuration. "
                "Please check OPENAI_API_BASE, OPENAI_API_KEY, and OPENAI_MODEL_NAME environment variables."
            )

        return ChatOpenAI(
            model=settings.openai_model_name,
            openai_api_base=settings.openai_api_base,
            openai_api_key=settings.openai_api_key,
            temperature=0.7,
            max_retries=2,
            timeout=30,
        )


if __name__ == "__main__":
    llm = LLMClient.create_llm()
    print(f"LLM Client created with model: {llm.model_name}")
