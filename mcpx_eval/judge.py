import logging
from typing import List
from datetime import datetime, timedelta
import json
import traceback

from mcpx_py import Chat, ChatConfig

from .models import Score, Results, Test, Model
from .database import Database
from .constants import SYSTEM_PROMPT, TEST_PROMPT

logger = logging.getLogger(__name__)


class Judge:
    model: str
    models: List[Model]
    ignore_tools: List[str]
    db: Database
    profile: str | None = None

    def __init__(
        self,
        models: List[Model | str] | None = None,
        db: Database | None = None,
        profile: str | None = None,
        judge_model: str = "claude-3-5-sonnet-latest",
        ignore_tools: List[str] | None = None,
    ):
        self.profile = profile or "~/default"
        self.ignore_tools = ignore_tools or []
        self.db = db or Database()
        self.models = []
        if models is not None:
            for model in models:
                self.add_model(model)

    def add_model(
        self,
        model: Model | str,
        profile: str | None = None,
    ):
        if isinstance(model, str):
            model = Model(
                name=model,
                config=ChatConfig(
                    model=model, system=TEST_PROMPT, ignore_tools=self.ignore_tools
                ),
            )
        else:
            model.config.ignore_tools.extend(self.ignore_tools)
        if model.config.client is None:
            model.config.client = self.agent.client
        model.config.max_tokens = 4096 * 2
        model.config.model = model.name
        self.models.append(model)

    async def run_test(self, test: Test, save=True) -> Results:
        results = await self.run(
            test.prompt,
            test.check,
            test.expected_tools,
        )
        if save:
            self.db.save_results(test.name, results)
        return results

    async def run(
        self,
        prompt,
        check,
        expected_tools,
    ) -> Results:
        m = []
        t = timedelta(seconds=0)
        for model in self.models:
            start = datetime.now()
            result = {"messages": []}
            logger.info(f"Evaluating model {model.slug}")
            try:
                chat = Chat(model.config)
                response, messages = await chat.inspect(prompt)
            except KeyboardInterrupt:
                continue
            except Exception:
                s = traceback.format_exc()
                logger.error(f"{model.slug} failed: {s}")
                continue
            tt = datetime.now() - start
            duration_seconds = tt.total_seconds()
            t += tt
            tool_calls = 0
            for msg in messages:
                print(msg)
                if hasattr(msg, "parts"):
                    for part in msg.parts:
                        if part.part_kind == "text":
                            result["messages"].append(
                                {"kind": part.part_kind, "text": part.content}
                            )
                        elif part.part_kind == "tool-call":
                            result["messages"].append(
                                {
                                    "kind": part.part_kind,
                                    "tool": {
                                        "name": part.tool_name,
                                        "input": part.args_as_dict(),
                                    },
                                    "tool_call_id": part.tool_call_id,
                                }
                            )
                            tool_calls += 1

            result["duration_in_seconds"] = f"{duration_seconds}s"
            result["number_of_tools_used"] = f"{tool_calls}"

            data = json.dumps(result)

            # Analyze tool usage
            tool_analysis = {}
            redundant_tool_calls = 0

            # Track previously seen tool patterns to detect redundancy
            seen_tool_patterns = set()

            # Process messages to analyze tool use
            for i, msg in enumerate(result["messages"]):
                if msg.get("tool"):
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
            agent = Chat(model.config, result_type=Score, system_prompt=SYSTEM_PROMPT)
            res = await agent.send_message(f"""
<settings>
Current date and time: {datetime.now().isoformat()}
</settings>
<prompt>
{prompt}
</prompt>
<output>
{data}
</output>
<check>{check}</check>
<expected-tools>{", ".join(expected_tools)}</expected-tools>
""")

            # Add additional metrics to the score
            score_data = res.data

            # Add tool analysis metrics and duration
            score_data.tool_analysis = tool_analysis
            score_data.redundant_tool_calls = redundant_tool_calls
            score_data.duration = duration_seconds
            score_data.model = model.slug

            m.append(score_data)
        return Results(scores=m, duration=t.total_seconds())
