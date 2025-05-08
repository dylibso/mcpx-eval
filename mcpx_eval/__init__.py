import logging

from .models import Score, Results, Model, Test, ModelConfig
from .database import Database
from .judge import Judge


from pydantic_ai.models.anthropic import (
    AnthropicModel as AnthropicModelConfig,
    AsyncAnthropic,
)
from pydantic_ai.models.openai import (
    OpenAIModel as OpenAIModelConfig,
    AsyncOpenAI,
)


class AnthropicClient(AsyncAnthropic):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)

    @property
    def client(self):
        return self._client

    @property
    def system(self):
        return self._system


class OpenAIClient(AsyncOpenAI):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)

    @property
    def client(self):
        return self._client

    @property
    def system(self):
        return self._system


logger = logging.getLogger(__name__)

__all__ = [
    "Score",
    "Results",
    "Model",
    "Test",
    "Database",
    "Judge",
    "ModelConfig",
    "AnthropicModelConfig",
    "OpenAIModelConfig",
    "AnthropicClient",
    "OpenAIClient",
]
