import json
import tomllib
import os
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List
import sqlite3

from mcpx_pydantic_ai import BaseModel, Agent, Field
from mcpx_py import Ollama, Claude, Gemini, OpenAI, ChatConfig, Chat

logger = logging.getLogger(__name__)


class Score(BaseModel):
    """
    Used to score the result of an LLM tool call
    """

    model: str = Field("Name of model being scored")
    duration: float = Field("Total time of call in seconds")
    llm_output: str = Field(
        "Model output, this is the 'content' field of the final message from the LLM"
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


class Database:
    conn: sqlite3.Connection

    def __init__(self, path: str = "eval.db"):
        self.conn = sqlite3.connect(path)

        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS tests (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                max_tool_calls INTEGER,
                prompt TEXT NOT NULL,
                prompt_check TEXT NOT NULL,
                UNIQUE(name)
            );
            CREATE TABLE IF NOT EXISTS eval_results (
                id INTEGER PRIMARY KEY,
                t TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                test_name TEXT NOT NULL,
                model TEXT NOT NULL,
                duration REAL NOT NULL,
                output TEXT NOT NULL,
                description TEXT NOT NULL,
                accuracy REAL NOT NULL,
                tool_use REAL NOT NULL,
                tool_calls INT NOT NULL,
                overall REAL NOT NULL,
                FOREIGN KEY(test_name) REFERENCES tests(name)
            );
        """
        )
        self.conn.commit()

    def save_score(self, name: str, score: Score, commit=True):
        if name == "":
            return
        self.conn.execute(
            """
                INSERT INTO eval_results (
                    test_name,
                    model,
                    duration,
                    output,
                    description,
                    accuracy,
                    tool_use,
                    tool_calls,
                    overall
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                name,
                score.model,
                score.duration,
                score.llm_output,
                score.description,
                score.accuracy,
                score.tool_use,
                score.tool_calls,
                score.overall,
            ),
        )
        if commit:
            self.conn.commit()

    def save_test(self, test: "Test"):
        self.conn.execute(
            """
            INSERT OR IGNORE INTO tests (name, max_tool_calls, prompt, prompt_check) VALUES (?, ?, ?, ?);
            """,
            (test.name, test.max_tool_calls, test.prompt, test.check),
        )
        self.conn.commit()

    def save_results(self, name: str, results: Results):
        for score in results.scores:
            self.save_score(name, score, commit=False)
        self.conn.commit()

    def average_results(self, name: str) -> Results:
        total_time = 0.0
        cursor = self.conn.execute(
            """
            SELECT t, model, duration, output, 
                   description, accuracy, tool_use, tool_calls, overall
            FROM eval_results 
            WHERE test_name=?
        """,
            (name,),
        )
        items = cursor.fetchall()
        out = []
        scoremap = {}
        for item in items:
            model = item[1]
            if model not in scoremap:
                scoremap[model] = []
            scoremap[model].append(
                Score(
                    model=model,
                    duration=item[2],
                    llm_output=item[3],
                    description=item[4],
                    accuracy=item[5],
                    tool_use=item[6],
                    tool_calls=item[7],
                    overall=item[8],
                )
            )

        for model, scores in scoremap.items():
            avg_duration = 0.0
            avg_accuracy = 0.0
            avg_tool_use = 0.0
            avg_tool_calls = 0.0
            avg_overall = 0.0
            description = []
            output = []
            for score in scores:
                avg_duration += score.duration
                avg_accuracy += score.accuracy
                avg_tool_use += score.tool_use
                avg_tool_calls += score.tool_calls
                avg_overall += score.overall
                description.append(score.description)
                output.append(score.llm_output)
            n = len(scores)
            out.append(
                Score(
                    model=model,
                    duration=avg_duration / n,
                    accuracy=avg_accuracy / n,
                    tool_use=avg_tool_use / n,
                    tool_calls=int(avg_tool_calls / n),
                    overall=avg_overall / n,
                    llm_output=f"Sample: {output[-1]}",
                    description=f"Sample: {description[-1]}",
                )
            )

        return Results(scores=out, duration=total_time)


class Judge:
    agent: Agent
    models: List[Model]
    db: Database

    def __init__(
        self, models: List[Model | str] | None = None, db: Database | None = None
    ):
        self.db = db or Database()
        self.agent = Agent(
            "claude-3-5-sonnet-latest", result_type=Score, system_prompt=SYSTEM_PROMPT
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

    async def run_test(self, test: "Test", save=True) -> Results:
        results = await self.run(
            test.prompt, test.check, max_tool_calls=test.max_tool_calls
        )
        if save:
            self.db.save_results(test.name, results)
        return results

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
                    model.config.system = """
                    You are a helpful large language model with tool calling access. Use the available tools
                    to determine results you cannot answer on your own
                    """
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

            data = json.dumps(result)

            logger.info(f"Analyzing results of {model.name}")
            res = await self.agent.run(
                user_prompt=f"<direction>Analyze the following results for the prompt {prompt}. {check}</direction>\n{data}"
            )
            m.append(res.data)
        return Results(scores=m, duration=t.total_seconds())


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
            data.get("models", []),
            max_tool_calls=data.get("max-tool-calls"),
        )
