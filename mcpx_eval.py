import json
from typing import List

from mcpx_pydantic_ai import BaseModel, Agent, Field
from mcpx_py import Ollama, ChatConfig


class Score(BaseModel):
    """
    Used to score the result of an LLM tool call
    """

    model: str = Field("Name of model being scored")
    output: str = Field("The literal output of the model being tested")
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
    def __init__(self, models: List[str] | None = None):
        if models is None:
            models = []
        self.agent = Agent(
            "claude-3-5-sonnet-latest", result_type=Scores, system_prompt=SYSTEM_PROMPT
        )
        self.models = models

    async def run(self, prompt, test) -> Scores:
        m = []

        for model in self.models:
            chat = Ollama(
                ChatConfig(
                    model=model,
                    system="Utilize tools when unable to determine a result on your own",
                )
            )
            chat.get_tools()
            result = {"model": model, "messages": []}
            async for response in chat.chat(prompt):
                tool = None
                if response.tool is not None:
                    tool = {"name": response.tool.name, "input": response.tool.input}
                result["messages"].append(
                    {
                        "content": response.content,
                        "role": response.role,
                        "is_error": response._error or False,
                        "tool": tool,
                    }
                )
            m.append(result)

        data = json.dumps(m)

        res = await self.agent.run(
            user_prompt=f"<direction>Analyze the following results for the prompt {prompt}. {test}</direction>\n{data}"
        )
        return res.data


async def main():
    judge = Judge(models=["llama3.2", "qwen2.5"])
    res = await judge.run(
        "how many images are there on google.com?",
        "the fetch tool should be used to determine there is only one image on google,com",
    )
    print(res)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
