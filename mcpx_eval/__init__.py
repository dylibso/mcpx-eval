
import json
import tomllib
import os
from typing import List

from mcpx_pydantic_ai import BaseModel, Agent, Field
from mcpx_py import Ollama, Claude, Gemini, OpenAI, ChatConfig


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


class Score(BaseModel):
    """
    Used to score the result of an LLM tool call
    """

    model: str = Field("Name of model being scored")
    output: str = Field("The message content from the final message sent by the model")
    description: str = Field("Description of results for this model")
    accuracy: float = Field("A score of how accurate the response is")
    tool_use: float = Field("A score of how appropriate the tool use is")
    overall: float = Field("An overall qualitative score of the response")


class Scores(BaseModel):
    scores: List[Score] = Field("A list of scores for each model")


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
"""


class Judge:
    def __init__(self, models: List[str] | None = None, log=print):
        if models is None:
            models = []
        self.agent = Agent(
            "claude-3-5-sonnet-latest", result_type=Scores, system_prompt=SYSTEM_PROMPT
        )
        self.models = models
        self.log = log

    async def run(self, prompt, check, max_tool_calls: int | None = None) -> Scores:
        m = []

        for model in self.models:
            self.log(f"Evaluating model {model}")
            if "claude" in model:
                chat = Claude(
                    ChatConfig(
                        model=model,
                    )
                )
            elif (
                model in ["gpt-4o", "o1", "o1-mini", "o3-mini", "o3"]
                or "gpt-3.5" in model
                or "gpt-4" in model
            ):
                chat = OpenAI(
                    ChatConfig(
                        model=model,
                    )
                )
            elif "gemini" in model:
                chat = Gemini(ChatConfig(model=model))
            else:
                chat = Ollama(
                    ChatConfig(
                        model=model,
                    )
                )
            chat.get_tools()
            result = {"model": model, "messages": []}
            tool_calls = 0
            try:
                async for response in chat.chat(prompt):
                    tool = None
                    if response.tool is not None:
                        tool = {
                            "name": response.tool.name,
                            "input": response.tool.input,
                        }
                        if max_tool_calls is not None and tool_calls >= max_tool_calls:
                            result["message"].append(
                                {
                                    "error": "Stopping, too many tool calls",
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
                self.log(f"Error: {str(exc)}")
                result["messages"].append(
                    {
                        "error": str(exc),
                        "role": "error",
                        "is_error": True,
                        "tool": None,
                    }
                )
            m.append(result)

        data = json.dumps(m)

        self.log("Analyzing results")
        res = await self.agent.run(
            user_prompt=f"<direction>Analyze the following results for the prompt {prompt}. {check}</direction>\n{data}"
        )
        return res.data


