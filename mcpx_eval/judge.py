import logging
from typing import List
from datetime import datetime, timedelta
import json
from mcpx_pydantic_ai import Agent
from mcpx_py import Chat, Claude, OpenAI, Gemini, Ollama, ChatConfig

from .models import Score, Results, Test, Model
from .database import Database
from .constants import SYSTEM_PROMPT, TEST_PROMPT

logger = logging.getLogger(__name__)


class Judge:
    agent: Agent
    model: str
    models: List[Model]
    db: Database
    profile: str | None

    def __init__(
        self,
        models: List[Model | str] | None = None,
        db: Database | None = None,
        profile: str | None = None,
        judge_model: str = "claude-3-5-sonnet-latest",
    ):
        self.profile = profile
        self.db = db or Database()
        self.agent = Agent(judge_model, result_type=Score, system_prompt=SYSTEM_PROMPT)
        if self.profile is not None:
            self.agent.client.set_profile(self.profile)
        self.models = []
        if models is not None:
            for model in models:
                self.add_model(model)

    def add_model(self, model: Model | str, profile: str | None = None):
        if isinstance(model, str):
            model = Model(
                name=model, config=ChatConfig(model=model, system=TEST_PROMPT)
            )
        model.config.model = model.name
        if model.config.client is None:
            model.config.client = self.agent.client
        if profile is not None:
            model.config.client.set_profile(profile)
        self.models.append(model)

    async def run_test(self, test: Test, save=True) -> Results:
        results = await self.run(
            test.prompt,
            test.check,
            test.expected_tools,
            max_tool_calls=test.max_tool_calls,
        )
        if save:
            self.db.save_results(test.name, results)
        return results

    async def run(
        self, prompt, check, expected_tools, max_tool_calls: int | None = None
    ) -> Results:
        m = []
        t = timedelta(seconds=0)
        model_cache = {}
        for model in self.models:
            logger.info(f"Evaluating model {model.name}")
            try:
                if model.name in model_cache:
                    chat = model_cache[model.slug]
                    chat.provider.clear_history()
                else:
                    if model.provider == "anthropic":
                        chat = Chat(Claude(config=model.config))
                    elif model.provider == "openai":
                        chat = Chat(OpenAI(config=model.config))
                    elif model.provider == "google":
                        chat = Chat(Gemini(config=model.config))
                    elif model.provider == "ollama":
                        chat = Chat(Ollama(config=model.config))
                    else:
                        logger.error(
                            f"Skipping invalid model provider: {model.provider}"
                        )
                    model_cache[model.name] = chat
                result = {"messages": []}
                tool_calls = 0
                chat.provider.config.client.clear_cache()
                start = datetime.now()
                async for response in chat.send_message(prompt):
                    tool = None
                    if response.tool is not None:
                        logger.info(f"Tool: {response.tool.name} {response.tool.input}")
                        logger.debug(f"Result: {response.content}")
                        tool = {
                            "name": response.tool.name,
                            "input": response.tool.input,
                        }
                        if max_tool_calls is not None and tool_calls >= max_tool_calls:
                            result["messages"].append(
                                {
                                    "error": f"Stopping, {tool_calls} tool calls when the maximum is {max_tool_calls}",
                                    "role": "error",
                                    "is_error": True,
                                    "tool": tool,
                                }
                            )
                            break
                        tool_calls += 1
                    result["messages"].append(
                        {
                            "content": response.content,
                            "role": response.role,
                            "is_error": response._error or False,
                            "tool": tool,
                        }
                    )
            except KeyboardInterrupt:
                continue
            except Exception as exc:
                logger.error(f"Error message: {str(exc)}")
                result["messages"].append(
                    {
                        "error": str(exc),
                        "role": "error",
                        "is_error": True,
                        "tool": None,
                    }
                )
            tt = datetime.now() - start
            duration_seconds = tt.total_seconds()
            t += tt
            result["duration"] = f"{duration_seconds}s"

            data = json.dumps(result)

            # Analyze tool usage
            tool_analysis = {}
            redundant_tool_calls = 0

            # Track previously seen tool patterns to detect redundancy
            seen_tool_patterns = set()

            # Process messages to analyze tool use
            for i, msg in enumerate(result["messages"]):
                if msg.get("tool") and not msg.get("is_error"):
                    tool_name = msg["tool"]["name"]
                    tool_input = msg["tool"]["input"]

                    # Create a pattern string for redundancy detection
                    tool_pattern = f"{tool_name}:{str(tool_input)}"

                    # Check for redundancy
                    if tool_pattern in seen_tool_patterns:
                        redundant_tool_calls += 1
                        redundancy_status = "redundant"
                    else:
                        seen_tool_patterns.add(tool_pattern)
                        redundancy_status = "unique"

                    # Store tool analysis
                    tool_analysis[f"tool_{i}"] = {
                        "name": tool_name,
                        "input": tool_input,
                        "redundancy": redundancy_status,
                    }

            logger.info(f"Analyzing results of {model.name}")
            if self.profile is None:
                self.agent.client.set_profile(model.profile)
            res = await self.agent.run(
                user_prompt=f"""
<settings>
Max tool calls: {max_tool_calls}
Current date and time: {datetime.now().isoformat()}
</settings>
<prompt>
{prompt}
</prompt>
<output>
{data}
</output>
<check>{check}</check>
<expected-tools>{", ".join(expected_tools)}</expected-tools>
"""
            )

            # Add additional metrics to the score
            score_data = res.data

            # Add tool analysis metrics and duration
            score_data.tool_analysis = tool_analysis
            score_data.redundant_tool_calls = redundant_tool_calls
            score_data.duration = duration_seconds
            score_data.model = model.slug

            m.append(score_data)
        return Results(scores=m, duration=t.total_seconds())
