from mcpx_pydantic_ai import BaseModel, Field
from typing import List
import pandas as pd
from dataclasses import dataclass
from mcpx_py import ChatConfig, client


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
    config: ChatConfig

    def __init__(self, name: str, config: ChatConfig):
        provider, n, profile = parse_model(name)
        self.provider = provider
        self.name = n
        self.profile = profile
        self.config = config
        self.config.client = client.Client(config=client.ClientConfig(profile=profile))

    @property
    def slug(self):
        if self.profile == "default" or self.profile == "~/default":
            return self.name
        if self.profile.startswith("~/"):
            return f"{self.name}/{self.profile.split('/', maxsplit=1)[1]}"
        return f"{self.name}/{self.profile}"


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
    tool_use: float = Field("A score (0-100) of how appropriate the tool use is")
    tool_calls: int = Field("Number of tool calls used")
    accuracy: float = Field(
        "A score (0-100) of how accurate the response is based on the output of the tool calls"
    )
    clarity: float = Field(
        "A score (0-100) of how clear and understandable the response is, this includes making sure the formatting is correct"
    )
    completeness: float = Field(
        "A score (0-100) of how complete the response is according to the check criteria"
    )
    quality: float = Field(
        "A score (0-100) of the overall quality of the response, this may include things not included in the other scores"
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

    # Detailed tool use analysis
    tool_analysis: dict = Field(
        {},
        description="Analysis of individual tool calls with success/relevance ratings",
    )
    redundant_tool_calls: int = Field(
        0, description="Number of redundant or unnecessary tool calls"
    )
    failed_tool_calls: int = Field(0, description="Number of failed tool calls")

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to a pandas DataFrame for analysis"""
        record = {
            "model": self.model,
            "duration": self.duration,
            "tool_use": self.tool_use,
            "tool_calls": self.tool_calls,
            "accuracy": self.accuracy,
            "clarity": self.clarity,
            "helpfulness": self.completeness,
            "overall": self.overall_quality,
            "hallucination_score": self.hallucination_score,
            "redundant_tool_calls": self.redundant_tool_calls,
            "false_claims_count": len(self.false_claims),
        }
        return pd.DataFrame(record)


class Results(BaseModel):
    scores: List[Score] = Field("A list of scores for each model")
    duration: float = Field("Total duration of all tests")

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to a pandas DataFrame for analysis"""
        records = []
        for score in self.scores:
            records.append(score.to_dataframe())
        return pd.concat(records)


class Test:
    name: str
    prompt: str
    check: str
    expected_tools: List[str]
    models: List[str]
    max_tool_calls: int | None

    def __init__(
        self,
        name: str,
        prompt: str,
        check: str,
        models: List[str],
        expected_tools: List[str],
        max_tool_calls: int | None = None,
    ):
        self.name = name
        self.prompt = prompt
        self.check = check
        self.models = models
        self.expected_tools = expected_tools
        self.max_tool_calls = max_tool_calls

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
                t.models = data.get("models", t.models)
                t.max_tool_calls = data.get("max-tool-calls", t.max_tool_calls)
                t.expected_tools.extend(data.get("expected-tools", []))
            return t
        return Test(
            data.get("name", path),
            data.get("prompt", ""),
            data.get("check", ""),
            data.get("models", []),
            data.get("expected-tools", []),
            max_tool_calls=data.get("max-tool-calls"),
        )
