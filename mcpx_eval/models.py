from mcpx_pydantic_ai import BaseModel, Field
from typing import List
import json

class Score(BaseModel):
    """
    Used to score the result of an LLM tool call
    """
    model: str = Field("Name of model being scored")
    duration: float = Field("Total time of call in seconds")
    llm_output: str = Field("Model output, this is the 'content' field of the final message from the LLM")
    description: str = Field("Description of results for this model")

    # Core metrics
    tool_use: float = Field("A score of how appropriate the tool use is")
    tool_calls: int = Field("Number of tool calls")
    accuracy: float = Field("A score of how accurate the response is")
    clarity: float = Field("A score of how clear and understandable the response is")
    helpfulness: float = Field("A score of how helpful the response is to the user")
    overall: float = Field("An overall score of the quality of the response, this may include things not included in the other scores")

    # Hallucination metrics
    hallucination_score: float = Field(0.0, description="A score representing the presence of hallucinations (lower is better)")
    false_claims: list = Field([], description="List of identified false claims or hallucinations in the response")

    # Detailed tool use analysis
    tool_analysis: dict = Field({}, description="Analysis of individual tool calls with success/relevance ratings")
    redundant_tool_calls: int = Field(0, description="Number of redundant or unnecessary tool calls")

class Results(BaseModel):
    scores: List[Score] = Field("A list of scores for each model")
    duration: float = Field("Total duration of all tests")

class Test:
    name: str
    prompt: str
    check: str
    models: List[str]
    max_tool_calls: int | None

    def __init__(self, name: str, prompt: str, check: str, models: List[str], max_tool_calls: int | None = None):
        self.name = name
        self.prompt = prompt
        self.check = check
        self.models = models
        self.max_tool_calls = max_tool_calls

    @staticmethod
    def load(path) -> "Test":
        import tomllib
        import os
        
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
