import logging
from typing import List
from datetime import datetime, timedelta
from dataclasses import dataclass
from mcpx_pydantic_ai import Agent
from mcpx_py import Chat, Claude, OpenAI, Gemini, Ollama, ChatConfig

from .models import Score, Results, Test
from .database import Database
from .constants import SYSTEM_PROMPT, TEST_PROMPT

logger = logging.getLogger(__name__)

@dataclass
class Model:
    name: str
    config: ChatConfig

class Judge:
    agent: Agent
    models: List[Model]
    db: Database

    def __init__(self, models: List[Model | str] | None = None, db: Database | None = None):
        self.db = db or Database()
        self.agent = Agent("claude-3-5-sonnet-latest", result_type=Score, system_prompt=SYSTEM_PROMPT)
        self.models = []
        if models is not None:
            for model in models:
                self.add_model(model)

    def add_model(self, model: Model | str):
        if isinstance(model, str):
            model = Model(name=model, config=ChatConfig(model=model))
        model.config.model = model.name
        if model.config.client is None:
            model.config.client = self.agent.client
        model.config.system = TEST_PROMPT
        self.models.append(model)

    async def run_test(self, test: Test, save=True) -> Results:
        results = await self.run(test.prompt, test.check, max_tool_calls=test.max_tool_calls)
        if save:
            self.db.save_results(test.name, results)
        return results

    async def run(self, prompt, check, max_tool_calls: int | None = None) -> Results:
        # ... [keeping the existing run implementation]
        pass
