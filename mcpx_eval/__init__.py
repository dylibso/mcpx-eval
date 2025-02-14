import json
import tomllib
import os
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List

from mcpx_pydantic_ai import BaseModel, Agent, Field
from mcpx_py import Ollama, Claude, Gemini, OpenAI, ChatConfig, Chat

logger = logging.getLogger(__name__)


class Score(BaseModel):
    """
    Used to score the result of an LLM tool call
    """

    model: str = Field("Name of model being scored")
    duration: float = Field("Total time of call in seconds")
    output: str = Field(
        "Resulting output of the prompt, taken from the 'content' field of the final message"
    )
    description: str = Field("Description of results for this model")
    accuracy: float = Field("A score of how accurate the response is")
    tool_use: float = Field("A score of how appropriate the tool use is")
    tool_calls: int = Field("Number of tool calls")
    overall: float = Field(
        "An overall score of the quality of the response, this may include things not included in the other scores"
    )


class Results(BaseModel):
    scores: List[Score] = Field("A list of scores for each model")
    duration: float = Field("Total duration of all tests")


SYSTEM_PROMPT = """
You are an large language model evaluator, you are an expert at comparing the output of various models based on 
accuracy, tool use and overall quality of the output.

- All numeric responses should be scored from 0.0 - 100.0, where 100 is the best score and 0 is the worst
- Additional direction for each evaluation may be marked in the input between <direction></direction> tags
- The tool use score should be based on whether or not the correct tool was used and whether the minimum amount
  of tools were used to accomplish a task. Over use of tools or repeated use of tools should deduct points from
  this score.
- The accuracy score should reflect the accuracy of the result generally and taking into account the <direction> block
- The overall score should reflect the overall quality of the output
- Try to utilize the tools that are available instead of searching for new tools
"""


@dataclass
class Model:
    name: str
    config: ChatConfig


class Judge:
    agent: Agent
    models: List[Model]

    def __init__(self, models: List[Model | str] | None = None):
        self.agent = Agent(
            "claude-3-5-sonnet-latest", result_type=Results, system_prompt=SYSTEM_PROMPT
        )
        self.models = []
        if models is not None:
            for model in models:
                self.add_model(model)

    def add_model(self, model: Model | str):
        if isinstance(model, str):
            model = Model(
                name=model,
                config=ChatConfig(
                    model=model,
                ),
            )
        model.config.model = model.name
        if model.config.client is None:
            model.config.client = self.agent.client
        self.models.append(model)

    async def run_test(self, test: "Test") -> Results:
        return await self.run(
            test.prompt, test.check, max_tool_calls=test.max_tool_calls
        )

    async def run(
        self,
        prompt,
        check,
        max_tool_calls: int | None = None,
    ) -> Results:
        m = []
        t = timedelta(seconds=0)
        model_cache = {}
        for model in self.models:
            logger.info(f"Evaluating model {model.name}")
            if model.name in model_cache:
                chat = model_cache[model.name]
            else:
                if "claude" in model.name:
                    chat = Chat(Claude(config=model.config))
                elif (
                    model.name in ["gpt-4o", "o1", "o1-mini", "o3-mini", "o3"]
                    or "gpt-3.5" in model.name
                    or "gpt-4" in model.name
                ):
                    chat = Chat(OpenAI(config=model.config))
                elif "gemini" in model.name:
                    chat = Chat(Gemini(config=model.config))
                else:
                    chat = Chat(Ollama(config=model.config))
                model_cache[model.name] = chat
            start = datetime.now()
            result = {"model": model.name, "messages": []}
            tool_calls = 0
            try:
                async for response in chat.send_message(prompt):
                    tool = None
                    if response.tool is not None:
                        logger.info(f"Tool: {response.tool.name}")
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
            t += tt
            result["duration"] = f"{t.total_seconds()}s"
            m.append(result)

        data = json.dumps(m)

        logger.info("Analyzing results")
        res = await self.agent.run(
            user_prompt=f"<direction>Analyze the following results for the prompt {prompt}. {check}</direction>\n{data}"
        )
        return res.data


class Test:
    name: str
    prompt: str
    check: str
    models: List[str]
    max_tool_calls: int | None

    def __init__(
        self,
        name: str,
        prompt: str,
        check: str,
        models: List[str],
        max_tool_calls: int | None = None,
    ):
        self.name = name
        self.prompt = prompt
        self.check = check
        self.models = models
        self.max_tool_calls = max_tool_calls

    @staticmethod
    def load(path) -> "Test":
        with open(path) as f:
            s = f.read()
        data = tomllib.loads(s)
        if "import" in data:
            t = Test.load(os.path.join(os.path.dirname(path), data["import"]))
            t.name = data.get("name", t.name)
            t.prompt = data.get("prompt", t.prompt)
            t.check = data.get("check", t.check)
            t.models = data.get("models", t.models)
            t.max_tool_calls = data.get("max-tool-calls", t.max_tool_calls)
            return t
        return Test(
            data.get("name", path),
            data["prompt"],
            data["check"],
            data["models"],
            max_tool_calls=data.get("max-tool-calls"),
        )
