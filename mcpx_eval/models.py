from mcpx_pydantic_ai import BaseModel, Field
from typing import List, Dict, Any
import pandas as pd
from dataclasses import dataclass


def parse_model(m: str) -> (str, str, str):
    if ":" in m:
        provider, name = m.split(":", maxsplit=1)
    else:
        name = m
        provider = None

    if "/" in name:
        name, profile = name.split("/", maxsplit=1)
        if "/" not in profile:
            profile = "~/" + profile
    else:
        profile = "~/default"

    if provider is None:
        if "claude" in name:
            provider = "anthropic"
        elif (
            name in ["gpt-4o", "o1", "o1-mini", "o3-mini", "o3"]
            or "gpt-3.5" in name
            or "gpt-4" in name
            or "gpt-4.5" in name
        ):
            provider = "openai"

        elif "gemini" in name:
            provider = "google"
        else:
            provider = "ollama"
    return (provider, name, profile)


@dataclass
class Model:
    name: str
    profile: str
    provider: str

    def __init__(self, name: str, profile: str | None = None):
        provider, n, p = parse_model(name)
        self.provider = provider
        self.name = n
        if profile is None:
            self.profile = p
        else:
            self.profile = profile

    @property
    def slug(self):
        if self.profile == "default" or self.profile == "~/default":
            return self.name
        if self.profile.startswith("~/"):
            return f"{self.name}/{self.profile.split('/', maxsplit=1)[1]}"
        return f"{self.name}/{self.profile}"

    @property
    def provider_and_name(self):
        return f"{self.provder}/{self.name}"


class ScoreModel(BaseModel):
    """
    Used to score the result of an LLM tool call
    """

    llm_output: str = Field(
        "",
        description="Model output, this is the 'content' field of the final message from the LLM",
    )
    description: str = Field("", description="Description of results for this model")

    # Core metrics
    tool_use: float = Field(
        0.0, description="A score (0-100) of how appropriate the tool use is"
    )
    accuracy: float = Field(
        0.0,
        description="A score (0-100) of how accurate the response is based on the output of the tool calls",
    )
    completeness: float = Field(
        0.0,
        description="A score (0-100) of how complete the response is according to the task at hand and <check> criteria",
    )
    quality: float = Field(
        0.0,
        description="A score (0-100) of the response quality - this includes the usefullness and clarity of the output",
    )

    # Hallucination metrics
    hallucination_score: float = Field(
        0.0,
        description="A score (0-100) representing the presence of hallucinations (lower is better)",
    )
    false_claims: list = Field(
        [],
        description="List of identified false claims or hallucinations in the response",
    )

    # Tools
    failed_tool_calls: int = Field(
        0,
        description="The number of failed tool calls, or tool calls that encountered an error",
    )


@dataclass
class Score:
    """
    Used to score the result of an LLM tool call
    """

    score: ScoreModel
    model: str
    duration: float

    # Detailed tool use analysis
    tool_analysis: dict
    redundant_tool_calls: int
    tool_calls: int

    def __getattribute__(self, name):
        if name == "score":
            return object.__getattribute__(self, name)
        if hasattr(self.score, name):
            return getattr(self.score, name)
        return object.__getattribute__(self, name)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to a pandas DataFrame for analysis"""
        record = {
            "model": self.model,
            "duration": self.duration,
            "tool_use": self.score.tool_use,
            "tool_calls": self.tool_calls,
            "accuracy": self.score.accuracy,
            "helpfulness": self.score.completeness,
            "quality": self.score.quality,
            "hallucination_score": self.store.hallucination_score,
            "redundant_tool_calls": self.redundant_tool_calls,
            "false_claims_count": len(self.score.false_claims),
        }
        return pd.DataFrame(record)


class Results(BaseModel):
    scores: List[Score] = Field([], description="A list of scores for each model")
    duration: float = Field(0.0, description="Total duration of all tests")

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to a pandas DataFrame for analysis"""
        records = []
        for score in self.scores:
            records.append(score.to_dataframe())
        return pd.concat(records)


class Test:
    name: str
    task: str | None
    prompt: str
    check: str
    expected_tools: List[str]
    ignore_tools: List[str]
    models: List[str]
    profile: str | None
    vars: Dict[str, Any]

    def __init__(
        self,
        name: str,
        prompt: str,
        check: str,
        models: List[str],
        expected_tools: List[str],
        ignore_tools: List[str] | None = None,
        profile: str | None = None,
        vars: Dict[str, Any] | None = None,
        task: str | None = None,
    ):
        self.name = name
        self.prompt = prompt
        self.check = check
        self.models = models
        self.expected_tools = expected_tools
        self.profile = profile
        self.ignore_tools = ignore_tools or []
        self.vars = vars or {}
        self.task = task

    @staticmethod
    def load(path) -> "Test":
        import tomllib
        import os

        with open(path) as f:
            s = f.read()
        data = tomllib.loads(s)
        if "import" in data:
            imports = data["import"]
            if isinstance(imports, str):
                imports = [imports]
            t = None
            for imp in imports:
                if t is None:
                    t = Test.load(os.path.join(os.path.dirname(path), imp))
                t.name = data.get("name", t.name)
                t.prompt = data.get("prompt", t.prompt)
                t.check = data.get("check", t.check)
                t.profile = data.get("profile", t.profile)
                t.models = data.get("models", t.models)
                t.expected_tools.extend(data.get("expected-tools", []))
                t.ignore_tools.extend(data.get("ignore-tools", []))
                t.vars.update(**data.get("ignore-tools", {}))
                t.task = t.task or data.get("task")
            return t
        return Test(
            data.get("name", path),
            data.get("prompt", ""),
            data.get("check", ""),
            data.get("models", []),
            data.get("expected-tools", []),
            ignore_tools=data.get("ignore-tools", []),
            vars=data.get("vars", {}),
            profile=data.get("profile", "~/default"),
            task=data.get("task"),
        )
