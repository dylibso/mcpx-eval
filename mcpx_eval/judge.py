import logging
from typing import List
from datetime import datetime, timedelta
import json
import traceback
import os

from mcpx_py import Chat, mcp_run, openai_compatible_model
import pystache

from .models import ScoreModel, Score, Results, Test, Model
from .database import Database
from .constants import SYSTEM_PROMPT, TEST_PROMPT

logger = logging.getLogger(__name__)


def mk_http(s):
    if not s.startswith("http"):
        return "http://" + s
    return s


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
        self.profile = profile or mcp_run.ProfileSlug("~", "default")
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
            )
        if profile is not None:
            model.profile = profile
        self.models.append(model)

    async def run_test(self, test: Test, save=True) -> Results:
        profile = test.profile
        if profile is None:
            profile = self.profile or mcp_run.ProfileSlug("~", "default")
        else:
            profile = mcp_run.ProfileSlug.parse(profile)
        if test.task is not None:
            client = mcp_run.Client(config=mcp_run.ClientConfig(profile=profile))
            tasks = client.tasks
            if test.task not in tasks:
                raise Exception(f"Invalid task, {test.task} not found in {profile}")
            test.prompt = tasks[test.task].prompt
        results = await self.run(
            pystache.render(test.prompt, test.vars),
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
            tool_calls = 0
            try:
                if model.provider == "ollama" or model.provider == "llama":
                    mx = openai_compatible_model(
                        mk_http(
                            os.environ.get(
                                f"{model.name.upper()}_HOST",
                                os.environ.get(
                                    "LLAMA_HOST",
                                    os.environ.get(
                                        "OLLAMA_HOST", "http://127.0.0.1:11434"
                                    ),
                                ),
                            )
                            + "/v1"
                        ),
                        model.name,
                    )
                elif model.provider == "openai":
                    mx = openai_compatible_model(
                        mk_http(
                            os.environ.get(
                                f"{model.name.upper()}_HOST",
                                os.environ.get("OPENAI_HOST", "https://api.openai.com"),
                            )
                            + "/v1",
                        ),
                        model.name,
                    )
                else:
                    mx = model.name
                chat = Chat(
                    client=mcp_run.Client(
                        config=mcp_run.ClientConfig(profile=model.profile)
                    ),
                    model=mx,
                    ignore_tools=self.ignore_tools,
                    system_prompt=TEST_PROMPT,
                    retries=5,
                )
                async for node in chat.iter(prompt):
                    if hasattr(node, "model_response"):
                        for part in node.model_response.parts:
                            if part.part_kind == "text":
                                logger.info(part.content)
                                result["messages"].append(
                                    {"kind": part.part_kind, "text": part.content}
                                )
                            elif part.part_kind == "tool-call":
                                logger.info(
                                    f"Tool {part.tool_name}({part.tool_call_id}): {part.args}"
                                )
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
                    elif hasattr(node, "request"):
                        for part in node.request.parts:
                            if part.part_kind == "text":
                                result["messages"].append(
                                    {"kind": part.part_kind, "text": part.content}
                                )
                            elif part.part_kind == "tool-return":
                                logger.info(
                                    f"Tool returned {part.tool_name}({part.tool_call_id})"
                                )
                                logger.debug(
                                    f"Tool result {part.tool_name}({part.tool_call_id}):\n{part.content}"
                                )
                                result["messages"].append(
                                    {
                                        "kind": part.part_kind,
                                        "tool_name": part.tool_name,
                                        "content": part.content,
                                        "tool_call_id": part.tool_call_id,
                                    }
                                )
                    elif hasattr(node, "data"):
                        logger.info(f"Final result: {node.data.data}")
                        result["messages"].append(
                            {"kind": "final_result", "text": node.data.data}
                        )
            except KeyboardInterrupt:
                continue
            except Exception:
                s = traceback.format_exc()
                logger.error(f"{model.slug} failed: {s}")
                continue
            tt = datetime.now() - start
            duration_seconds = tt.total_seconds()
            t += tt

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

            logger.info(f"Analyzing results of {model.slug}")
            agent = Chat(
                client=mcp_run.Client(
                    config=mcp_run.ClientConfig(profile=model.profile)
                ),
                model=model.name,
                ignore_tools=self.ignore_tools,
                result_type=ScoreModel,
                system_prompt=SYSTEM_PROMPT,
                result_retries=10,
            )
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

            m.append(
                Score(
                    score=res.data,
                    model=model.slug,
                    duration=duration_seconds,
                    tool_analysis=tool_analysis,
                    redundant_tool_calls=redundant_tool_calls,
                    tool_calls=tool_calls,
                )
            )
        return Results(scores=m, duration=t.total_seconds())
