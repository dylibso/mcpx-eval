from pydantic import BaseModel, Field
from pydantic_ai.models import Model as ModelConfig
from typing import List, Dict, Any, Tuple, Optional
import pandas as pd
from dataclasses import dataclass
from .constants import OPENAI_MODELS, DEFAULT_PROFILE


import json


def normalize_profile(profile: str) -> str:
    """Normalize a profile path to ensure it has the proper format."""
    if not profile:
        return DEFAULT_PROFILE
    if not profile.startswith("~/"):
        return "~/" + profile
    return profile


def parse_model(m: str) -> Tuple[Optional[str], str, str]:
    """Parse a model string into provider, name and profile components."""
    provider = None
    name = m
    profile = DEFAULT_PROFILE

    if not isinstance(m, str):
        provider = m.system
        model_name = m.model_name
        return (provider, model_name, profile)

    # Split provider and name
    if ":" in m:
        provider, name = m.split(":", maxsplit=1)

    # Split name and profile
    if "/" in name:
        name, profile = name.split("/", maxsplit=1)
        profile = normalize_profile(profile)

    # Infer provider if not specified
    if provider is None:
        if "claude" in name:
            provider = "anthropic"
        elif any(model in name for model in OPENAI_MODELS):
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
    trace: dict | None = None
    model_config: ModelConfig | None = None

    def __init__(
        self,
        name: str,
        profile: Optional[str] = None,
        trace: dict | None = None,
        model_config: ModelConfig | None = None,
    ):
        self.model_config = model_config
        if model_config is not None:
            provider, model_name, prof = parse_model(self.model_config)
        else:
            provider, model_name, prof = parse_model(name)
        self.provider = provider
        self.name = model_name
        self.profile = profile if profile is not None else prof
        self.trace = trace

    @property
    def slug(self) -> str:
        """Generate a slug identifier for the model."""
        if self.profile in [DEFAULT_PROFILE, "default"]:
            return self.name
        if self.profile.startswith("~/"):
            return f"{self.name}/{self.profile.split('/', maxsplit=1)[1]}"
        return f"{self.name}/{self.profile}"

    @property
    def provider_and_name(self) -> str:
        """Generate the provider/name identifier."""
        return f"{self.provider}/{self.name}"

    @staticmethod
    def load_trace(path):
        """Load trace from disk"""
        with open(path, "r") as f:
            data = json.load(f)
            model = data.pop("model")
            return Model(model, trace=data)


class ScoreModel(BaseModel):
    """Used to score the result of an LLM tool call."""

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
    """Used to score the result of an LLM tool call."""

    score: ScoreModel
    model: str
    duration: float
    tool_analysis: dict
    redundant_tool_calls: int
    tool_calls: int
    trace: dict | None = None

    def __getattribute__(self, name: str) -> Any:
        if name == "score":
            return object.__getattribute__(self, name)
        if hasattr(self.score, name):
            return getattr(self.score, name)
        return object.__getattribute__(self, name)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to a pandas DataFrame for analysis."""
        record = {
            "model": self.model,
            "duration": self.duration,
            "tool_use": self.score.tool_use,
            "tool_calls": self.tool_calls,
            "accuracy": self.score.accuracy,
            "helpfulness": self.score.completeness,
            "quality": self.score.quality,
            "hallucination_score": self.score.hallucination_score,
            "redundant_tool_calls": self.redundant_tool_calls,
            "false_claims_count": len(self.score.false_claims),
            "trace": self.trace,
        }
        return pd.DataFrame(record)

    def save_trace(self, path):
        """Save trace to disk"""
        trace = self.trace.copy()
        trace["model"] = self.model
        with open(path, "w") as f:
            f.write(json.dumps(trace))


class Results(BaseModel):
    """Collection of scores from multiple model evaluations."""

    scores: List[Score] = Field([], description="A list of scores for each model")
    duration: float = Field(0.0, description="Total duration of all tests")

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to a pandas DataFrame for analysis."""
        records = []
        for score in self.scores:
            records.append(score.to_dataframe())
        return pd.concat(records)


@dataclass
class Test:
    """Configuration for a model evaluation test."""

    name: str
    prompt: str
    check: str
    models: List[str]
    expected_tools: List[str]
    ignore_tools: List[str]
    profile: Optional[str]
    vars: Dict[str, Any]
    task: Optional[str]
    task_run: Optional[str]

    def __init__(
        self,
        name: str,
        prompt: str,
        check: str = "",
        models: List[str] | None = None,
        expected_tools: List[str] | None = None,
        ignore_tools: Optional[List[str]] = None,
        profile: Optional[str] = None,
        vars: Optional[Dict[str, Any]] = None,
        task: Optional[str] = None,
        task_run: Optional[str] = None,
    ):
        self.name = name
        self.prompt = prompt
        self.check = check
        self.models = models or []
        self.expected_tools = expected_tools or []
        self.profile = profile
        self.ignore_tools = ignore_tools or []
        self.vars = vars or {}
        self.task = task
        self.task_run = task_run

    @staticmethod
    def from_dict(data: dict) -> "Test":
        """Parse a dict into a test"""
        return Test(
            data.get("name", ""),
            data.get("prompt", ""),
            data.get("check", ""),
            data.get("models", []),
            data.get("expected-tools", []),
            ignore_tools=data.get("ignored-tools", data.get("ignore-tools", [])),
            vars=data.get("vars", {}),
            profile=data.get("profile"),
            task=data.get("task"),
            task_run=data.get("task-run"),
        )

    @staticmethod
    def load(path: str) -> "Test":
        """Load a test configuration from a TOML file."""
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

                # Update test attributes with any overrides from current file
                t.name = data.get("name", t.name)
                t.prompt = data.get("prompt", t.prompt)
                t.check = data.get("check", t.check)
                t.profile = data.get("profile", t.profile)
                t.models = data.get("models", t.models)
                t.expected_tools.extend(data.get("expected-tools", []))
                t.ignore_tools.extend(
                    data.get("ignored-tools", data.get("ignore-tools", []))
                )
                t.vars.update(**data.get("vars", {}))
                t.task = t.task or data.get("task")
                t.task_run = t.task_run or data.get("task-run")
            return t

        if "name" not in data:
            data["name"] = path

        return Test.from_dict(data)
