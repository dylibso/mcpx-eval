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

    # Core metrics
    tool_use: float = Field("A score of how appropriate the tool use is")
    tool_calls: int = Field("Number of tool calls")
    accuracy: float = Field("A score of how accurate the response is")
    clarity: float = Field("A score of how clear and understandable the response is")
    helpfulness: float = Field("A score of how helpful the response is to the user")
    overall: float = Field(
        "An overall score of the quality of the response, this may include things not included in the other scores"
    )

    # Hallucination metrics
    hallucination_score: float = Field(
        0.0,
        description="A score representing the presence of hallucinations (lower is better)",
    )
    false_claims: list = Field(
        [],
        description="List of identified false claims or hallucinations in the response",
    )

    # Detailed tool use analysis
    tool_analysis: dict = Field(
        {},
        description="Analysis of individual tool calls with success/relevance ratings",
    )
    redundant_tool_calls: int = Field(
        0, description="Number of redundant or unnecessary tool calls"
    )


class Results(BaseModel):
    scores: List[Score] = Field("A list of scores for each model")
    duration: float = Field("Total duration of all tests")


SYSTEM_PROMPT = """
You are a large language model evaluator, you are an expert at comparing the output of various models based on 
accuracy, tool use, user experience, and overall quality of the output.

- All numeric responses should be scored from 0.0 - 100.0, where 100 is the best score and 0 is the worst
- Additional direction for each evaluation may be marked in the input between <direction></direction> tags
- Do not make assumptions about improvments to the quality of the output beyond what is noted in the <check></check> tags

Performance metrics:
- The accuracy score should reflect the accuracy of the result generally and taking into account the <direction> block
- The tool use score should be based on whether or not the correct tool was used and whether the minimum amount
  of tools were used to accomplish a task. Over use of tools or repeated use of tools should deduct points from
  this score.

User-perceived quality metrics:
- The clarity score should measure how clear, concise, and understandable the model's response is
- The helpfulness score should measure how useful the response is in addressing the user's need

Advanced evaluation metrics:
- The hallucination_score should measure the presence of made-up, incorrect, or factually unsupported statements
  (lower is better, with 0 being no hallucinations and 100 being completely hallucinated)
- hallucination_score should only apply to made up information, if information is true at the time of the request
  it should be considered to be true
- The false_claims field should list any specific false statements or hallucinations identified in the response

- The overall score should reflect the overall quality of the output, considering both performance and user experience
- Try to utilize the tools that are available instead of searching for new tools

For responses containing hallucinations, analyze:
1. The severity of each hallucination (minor factual error vs completely fabricated information)
2. The confidence with which hallucinated content is presented
3. Whether hallucinations are central to the response or peripheral
4. Whether the hallucination could lead to harmful actions if believed

Be thorough in your evaluation, considering how well the model's response meets both technical requirements and user needs.
"""

TEST_PROMPT = """
You are a helpful AI assistant with access to various external tools and APIs. Your goal is to complete tasks thoroughly and autonomously by making full use of these tools. Here are your core operating principles:

1. Take initiative - Don't wait for user permission to use tools. If a tool would help complete the task, use it immediately.
2. Chain multiple tools together - Many tasks require multiple tool calls in sequence. Plan out and execute the full chain of calls needed to achieve the goal.
3. Handle errors gracefully - If a tool call fails, try alternative approaches or tools rather than asking the user what to do.
4. Make reasonable assumptions - When tool calls require parameters, use your best judgment to provide appropriate values rather than asking the user.
5. Show your work - After completing tool calls, explain what you did and show relevant results, but focus on the final outcome the user wanted.
6. Be thorough - Use tools repeatedly as needed until you're confident you've fully completed the task. Don't stop at partial solutions.
7. Try to utilize the tools that are available instead of searching for new tools

Your responses should focus on results rather than asking questions. Only ask the user for clarification if the task itself is unclear or impossible with the tools available.
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
                redundant_tool_calls INT NOT NULL DEFAULT 0,
                clarity REAL NOT NULL DEFAULT 0.0,
                helpfulness REAL NOT NULL DEFAULT 0.0, 
                overall REAL NOT NULL,
                hallucination_score REAL NOT NULL DEFAULT 0.0,
                false_claims TEXT NOT NULL DEFAULT '[]',
                tool_analysis TEXT NOT NULL DEFAULT '{}',
                FOREIGN KEY(test_name) REFERENCES tests(name)
            );
            
            -- Visualization table removed
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
                    redundant_tool_calls,
                    clarity,
                    helpfulness,
                    overall,
                    hallucination_score,
                    false_claims,
                    tool_analysis
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
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
                score.redundant_tool_calls,
                score.clarity,
                score.helpfulness,
                score.overall,
                score.hallucination_score,
                json.dumps(score.false_claims),
                json.dumps(score.tool_analysis),
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

    # Visualization methods removed
        
    def generate_json_summary(self):
        """
        Generate a comprehensive JSON summary of all test data in the database
        
        Returns:
            dict: A JSON-serializable dictionary containing all test results
        """
        # Get all test names
        cursor = self.conn.execute("SELECT name FROM tests")
        test_names = [row[0] for row in cursor.fetchall()]
        
        summary = {
            "tests": {},
            "total": {
                "models": {},
                "metrics": {
                    "accuracy": 0.0,
                    "tool_use": 0.0,
                    "clarity": 0.0,
                    "helpfulness": 0.0,
                    "overall": 0.0,
                    "hallucination_score": 0.0,
                    "tool_calls": 0,
                    "redundant_tool_calls": 0
                },
                "test_count": len(test_names),
                "model_count": 0
            }
        }
        
        all_models = set()
        
        # Process each test
        for test_name in test_names:
            results = self.average_results(test_name)
            
            test_summary = {
                "models": {},
                "metrics": {
                    "accuracy": 0.0,
                    "tool_use": 0.0,
                    "clarity": 0.0,
                    "helpfulness": 0.0,
                    "overall": 0.0,
                    "hallucination_score": 0.0,
                    "tool_calls": 0,
                    "redundant_tool_calls": 0
                },
                "model_count": len(results.scores)
            }
            
            # Process each model's score for this test
            for score in results.scores:
                model_name = score.model
                all_models.add(model_name)
                
                # Add model data to test summary
                test_summary["models"][model_name] = {
                    "accuracy": score.accuracy,
                    "tool_use": score.tool_use,
                    "clarity": score.clarity,
                    "helpfulness": score.helpfulness,
                    "overall": score.overall,
                    "hallucination_score": score.hallucination_score,
                    "tool_calls": score.tool_calls,
                    "redundant_tool_calls": score.redundant_tool_calls,
                    "duration": score.duration,
                    "false_claims": score.false_claims
                }
                
                # Update test metrics totals
                test_summary["metrics"]["accuracy"] += score.accuracy
                test_summary["metrics"]["tool_use"] += score.tool_use
                test_summary["metrics"]["clarity"] += score.clarity
                test_summary["metrics"]["helpfulness"] += score.helpfulness
                test_summary["metrics"]["overall"] += score.overall
                test_summary["metrics"]["hallucination_score"] += score.hallucination_score
                test_summary["metrics"]["tool_calls"] += score.tool_calls
                test_summary["metrics"]["redundant_tool_calls"] += score.redundant_tool_calls
            
            # Calculate test averages
            if test_summary["model_count"] > 0:
                for metric in test_summary["metrics"]:
                    if metric in ["tool_calls", "redundant_tool_calls"]:
                        # For integer metrics, we keep the sum
                        continue
                    test_summary["metrics"][metric] /= test_summary["model_count"]
            
            # Add test to summary
            summary["tests"][test_name] = test_summary
        
        # Update total metrics
        summary["total"]["model_count"] = len(all_models)
        
        # Initialize model data in total summary
        for model in all_models:
            summary["total"]["models"][model] = {
                "accuracy": 0.0,
                "tool_use": 0.0,
                "clarity": 0.0,
                "helpfulness": 0.0,
                "overall": 0.0,
                "hallucination_score": 0.0,
                "tool_calls": 0,
                "redundant_tool_calls": 0,
                "test_count": 0,
                "duration": 0.0
            }
        
        # Calculate totals across all tests
        for test_name, test_data in summary["tests"].items():
            for model, model_data in test_data["models"].items():
                summary["total"]["models"][model]["accuracy"] += model_data["accuracy"]
                summary["total"]["models"][model]["tool_use"] += model_data["tool_use"]
                summary["total"]["models"][model]["clarity"] += model_data["clarity"]
                summary["total"]["models"][model]["helpfulness"] += model_data["helpfulness"]
                summary["total"]["models"][model]["overall"] += model_data["overall"]
                summary["total"]["models"][model]["hallucination_score"] += model_data["hallucination_score"]
                summary["total"]["models"][model]["tool_calls"] += model_data["tool_calls"]
                summary["total"]["models"][model]["redundant_tool_calls"] += model_data["redundant_tool_calls"]
                summary["total"]["models"][model]["duration"] += model_data["duration"]
                summary["total"]["models"][model]["test_count"] += 1
        
        # Calculate model averages in total
        for model, model_data in summary["total"]["models"].items():
            test_count = model_data["test_count"]
            if test_count > 0:
                for metric in ["accuracy", "tool_use", "clarity", "helpfulness", "overall", "hallucination_score", "duration"]:
                    model_data[metric] /= test_count
        
        # Calculate overall metrics averages
        if summary["total"]["test_count"] > 0:
            for test_name, test_data in summary["tests"].items():
                summary["total"]["metrics"]["accuracy"] += test_data["metrics"]["accuracy"]
                summary["total"]["metrics"]["tool_use"] += test_data["metrics"]["tool_use"]
                summary["total"]["metrics"]["clarity"] += test_data["metrics"]["clarity"]
                summary["total"]["metrics"]["helpfulness"] += test_data["metrics"]["helpfulness"]
                summary["total"]["metrics"]["overall"] += test_data["metrics"]["overall"]
                summary["total"]["metrics"]["hallucination_score"] += test_data["metrics"]["hallucination_score"]
                summary["total"]["metrics"]["tool_calls"] += test_data["metrics"]["tool_calls"]
                summary["total"]["metrics"]["redundant_tool_calls"] += test_data["metrics"]["redundant_tool_calls"]
            
            for metric in ["accuracy", "tool_use", "clarity", "helpfulness", "overall", "hallucination_score"]:
                summary["total"]["metrics"][metric] /= summary["total"]["test_count"]
        
        # Add timestamp
        from datetime import datetime
        summary["generated_at"] = datetime.now().isoformat()
        
        return summary

    def average_results(self, name: str) -> Results:
        total_time = 0.0
        cursor = self.conn.execute(
            """
            SELECT t, model, duration, output, 
                   description, accuracy, tool_use, tool_calls, 
                   redundant_tool_calls, clarity, helpfulness, 
                   overall, hallucination_score, false_claims,
                   tool_analysis
            FROM eval_results 
            WHERE test_name LIKE ?
        """,
            (name.replace('*', '%'),),
        )
        items = cursor.fetchall()
        out = []
        scoremap = {}
        for item in items:
            model = item[1]
            model_key = model

            # Parse JSON fields
            false_claims = json.loads(item[13]) if item[13] else []
            tool_analysis = json.loads(item[14]) if item[14] else {}

            if model_key not in scoremap:
                scoremap[model_key] = []
            try:
                duration = float(item[2])  # Try to convert duration to float
            except (ValueError, TypeError):
                duration = 0.0  # Default to 0 if conversion fails
                
            scoremap[model_key].append(
                Score(
                    model=model,
                    duration=duration,
                    llm_output=item[3],
                    description=item[4],
                    accuracy=item[5],
                    tool_use=item[6],
                    tool_calls=item[7],
                    redundant_tool_calls=item[8],
                    clarity=item[9],
                    helpfulness=item[10],
                    overall=item[11],
                    hallucination_score=item[12],
                    false_claims=false_claims,
                    tool_analysis=tool_analysis,
                )
            )

        for model_key, scores in scoremap.items():
            # Calculate averages
            avg_duration = sum(float(score.duration) for score in scores)
            avg_accuracy = 0.0
            avg_tool_use = 0.0
            avg_tool_calls = 0.0
            avg_redundant_tool_calls = 0.0
            avg_clarity = 0.0
            avg_helpfulness = 0.0
            avg_overall = 0.0
            avg_hallucination = 0.0

            # Lists for calculating standard deviations
            all_accuracy = []
            all_overall = []

            # Collect data for aggregation
            description = []
            output = []
            combined_false_claims = []
            combined_tool_analysis = {}

            model = scores[0].model

            for score in scores:
                avg_accuracy += score.accuracy
                avg_tool_use += score.tool_use
                avg_tool_calls += score.tool_calls
                avg_redundant_tool_calls += score.redundant_tool_calls
                avg_clarity += score.clarity
                avg_helpfulness += score.helpfulness
                avg_overall += score.overall
                avg_hallucination += score.hallucination_score

                # Collect for std deviation calculation
                all_accuracy.append(score.accuracy)
                all_overall.append(score.overall)

                # Aggregate data
                description.append(score.description)
                output.append(score.llm_output)

                # Merge unique false claims
                for claim in score.false_claims:
                    if claim not in combined_false_claims:
                        combined_false_claims.append(claim)

                # Merge tool analysis data
                for tool_id, analysis in score.tool_analysis.items():
                    if tool_id not in combined_tool_analysis:
                        combined_tool_analysis[tool_id] = []
                    combined_tool_analysis[tool_id].append(analysis)

            # Calculate standard deviations
            # accuracy_std = (
            #     statistics.stdev(all_accuracy) if len(all_accuracy) > 1 else 0.0
            # )
            # overall_std = statistics.stdev(all_overall) if len(all_overall) > 1 else 0.0

            n = len(scores)
            out.append(
                Score(
                    model=model,
                    duration=avg_duration / n,
                    accuracy=avg_accuracy / n,
                    tool_use=avg_tool_use / n,
                    tool_calls=int(avg_tool_calls / n),
                    redundant_tool_calls=int(avg_redundant_tool_calls / n),
                    clarity=avg_clarity / n,
                    helpfulness=avg_helpfulness / n,
                    overall=avg_overall / n,
                    hallucination_score=avg_hallucination / n,
                    false_claims=combined_false_claims,
                    tool_analysis=combined_tool_analysis,
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
        model.config.system = TEST_PROMPT
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
            res = await self.agent.run(
                user_prompt=f"""<direction>
Analyze the following results for the prompt {prompt}.

For the hallucination_score metric (0-100 scale, lower is better), carefully check for any false statements, 
incorrect information, or made-up facts in the response and list them in the false_claims field.
</direction>
<check>{check}</check>
{data}"""
            )

            # Add additional metrics to the score
            score_data = res.data

            # Add tool analysis metrics and duration
            score_data.tool_analysis = tool_analysis
            score_data.redundant_tool_calls = redundant_tool_calls
            score_data.duration = duration_seconds

            m.append(score_data)
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
