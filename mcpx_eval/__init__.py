import logging
import sqlite3
import tomllib
import os
from datetime import datetime, timedelta
import json
from typing import List
import pandas as pd

from mcpx_pydantic_ai import Agent
from mcpx_py import Ollama, Claude, Gemini, OpenAI, ChatConfig, Chat

from .models import Score, Results, Model, Test
from .database import Database
from .constants import SYSTEM_PROMPT, TEST_PROMPT

logger = logging.getLogger(__name__)

__all__ = ['Score', 'Results', 'Model', 'Test', 'Database', 'Judge']

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
        Generate a comprehensive JSON summary of all test data in the database using pandas

        Returns:
            dict: A JSON-serializable dictionary containing all test results
        """
        # Get all results into a DataFrame
        df = pd.read_sql_query(
            """
            SELECT 
                test_name,
                model,
                duration,
                accuracy,
                tool_use,
                tool_calls,
                redundant_tool_calls,
                clarity,
                helpfulness,
                overall,
                hallucination_score,
                false_claims
            FROM eval_results
            """,
            self.conn
        )
        
        if df.empty:
            return {"tests": {}, "total": {"models": {}, "metrics": {}, "test_count": 0, "model_count": 0}}
            
        # Convert false_claims from JSON string
        df['false_claims'] = df['false_claims'].apply(json.loads)
        
        # Calculate test-level metrics
        test_metrics = df.groupby('test_name').agg({
            'model': 'nunique',  # Count unique models per test
            'accuracy': 'mean',
            'tool_use': 'mean',
            'tool_calls': 'sum',
            'redundant_tool_calls': 'sum',
            'clarity': 'mean',
            'helpfulness': 'mean',
            'overall': 'mean',
            'hallucination_score': 'mean'
        }).rename(columns={'model': 'model_count'})
        
        # Calculate model-level metrics for each test
        model_metrics = df.groupby(['test_name', 'model']).agg({
            'duration': 'mean',
            'accuracy': 'mean',
            'tool_use': 'mean',
            'tool_calls': 'sum',
            'redundant_tool_calls': 'sum',
            'clarity': 'mean',
            'helpfulness': 'mean',
            'overall': 'mean',
            'hallucination_score': 'mean',
            'false_claims': lambda x: list(set([item for sublist in x for item in sublist]))
        }).reset_index()
        
        # Calculate total metrics across all tests
        total_metrics = df.agg({
            'accuracy': 'mean',
            'tool_use': 'mean',
            'tool_calls': 'sum',
            'redundant_tool_calls': 'sum',
            'clarity': 'mean',
            'helpfulness': 'mean',
            'overall': 'mean',
            'hallucination_score': 'mean'
        }).to_dict()
        
        # Calculate total model metrics
        total_model_metrics = df.groupby('model').agg({
            'test_name': 'nunique',  # Count tests per model
            'duration': 'mean',
            'accuracy': 'mean',
            'tool_use': 'mean',
            'tool_calls': 'sum',
            'redundant_tool_calls': 'sum',
            'clarity': 'mean',
            'helpfulness': 'mean',
            'overall': 'mean',
            'hallucination_score': 'mean'
        }).rename(columns={'test_name': 'test_count'})
        
        # Build summary dictionary
        summary = {
            "tests": {},
            "total": {
                "models": {},
                "metrics": total_metrics,
                "test_count": len(test_metrics),
                "model_count": df['model'].nunique()
            },
            "generated_at": datetime.now().isoformat()
        }
        
        # Add test summaries
        for test_name in test_metrics.index:
            test_models = model_metrics[model_metrics['test_name'] == test_name]
            summary["tests"][test_name] = {
                "models": {
                    row['model']: {
                        "accuracy": row['accuracy'],
                        "tool_use": row['tool_use'],
                        "clarity": row['clarity'],
                        "helpfulness": row['helpfulness'],
                        "overall": row['overall'],
                        "hallucination_score": row['hallucination_score'],
                        "tool_calls": int(row['tool_calls']),
                        "redundant_tool_calls": int(row['redundant_tool_calls']),
                        "duration": row['duration'],
                        "false_claims": row['false_claims']
                    }
                    for _, row in test_models.iterrows()
                },
                "metrics": test_metrics.loc[test_name].to_dict(),
                "model_count": int(test_metrics.loc[test_name, 'model_count'])
            }
        
        # Add total model metrics
        for model in total_model_metrics.index:
            summary["total"]["models"][model] = {
                **total_model_metrics.loc[model].to_dict(),
                "test_count": int(total_model_metrics.loc[model, 'test_count'])
            }
            
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
            (name.replace("*", "%"),),
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
                        logger.info(f"Tool: {response.tool.name} {response.tool.input}")
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
