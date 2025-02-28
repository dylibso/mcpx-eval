from . import Judge, Test, Database
from .html import visualize_json
import asyncio
import logging
import pandas as pd
from tempfile import NamedTemporaryFile
import webbrowser

logger = logging.getLogger(__name__)


def print_result(result):
    # Print model header
    print(f"\n{result.model}")
    print("=" * len(result.model))

    # Create a DataFrame for the metrics
    metrics_df = pd.DataFrame(
        {
            "Metric": [
                "Duration (s)",
                "Tool Calls",
                "Redundant Calls",
                "Failed Calls",
                "Tool Use %",
                "Accuracy %",
                "Completeness %",
                "Quality %",
                "Hallucination Score",
            ],
            "Value": [
                f"{result.duration:.2f}",
                result.tool_calls,
                result.redundant_tool_calls,
                result.failed_tool_calls,
                f"{result.tool_use:.1f}",
                f"{result.accuracy:.1f}",
                f"{result.completeness:.1f}",
                f"{result.quality:.1f}",
                f"{result.hallucination_score:.1f}",
            ],
        }
    )

    # Print metrics table
    print("\nMetrics:")
    print(metrics_df.to_string(index=False))

    # Print output and description
    print("\nOutput:")
    print(result.llm_output)
    print("\nDescription:")
    print(result.description)

    # Print false claims if any
    if result.false_claims and len(result.false_claims) > 0:
        print("\nFalse Claims Detected:")
        for claim in result.false_claims:
            print(f"  - {claim}")

    # Print tool analysis if any
    if result.tool_analysis and len(result.tool_analysis) > 0:
        print("\nTool Analysis:")
        tool_data = []
        for tool_id, analysis in result.tool_analysis.items():
            if isinstance(analysis, list):
                for a in analysis:
                    tool_data.append(
                        {
                            "Tool ID": tool_id,
                            "Name": a["name"],
                            "Redundancy": a["redundancy"],
                        }
                    )
            else:
                tool_data.append(
                    {
                        "Tool ID": tool_id,
                        "Name": analysis["name"],
                        "Redundancy": analysis["redundancy"],
                    }
                )

        if tool_data:
            tool_df = pd.DataFrame(tool_data)
            print(tool_df.to_string(index=False))


def summary(args):
    db = Database()
    res = db.average_results(args.name)
    if not res.scores:
        return  # Database class now handles empty results messaging
    
    print(f"\nTest Summary: {args.name}")
    print("=" * (14 + len(args.name)))
    print(f"Number of results: {len(res.scores)}\n")
    
    for result in res.scores:
        print_result(result)


def json_summary(args):
    """Generate a JSON summary of test data"""
    import json

    db = Database()
    summary = db.generate_json_summary()

    # Filter to specific test if requested
    if args.name:
        if args.name in summary["tests"]:
            filtered_summary = {
                "tests": {args.name: summary["tests"][args.name]},
                "total": {
                    "models": {},
                    "metrics": summary["tests"][args.name]["metrics"],
                    "test_count": 1,
                    "model_count": summary["tests"][args.name]["model_count"],
                },
                "generated_at": summary["generated_at"],
            }
            # Include only models that participated in this test
            for model_name, model_data in summary["total"]["models"].items():
                if model_name in summary["tests"][args.name]["models"]:
                    filtered_summary["total"]["models"][model_name] = {
                        **model_data,
                        "test_count": 1,
                    }
            summary = filtered_summary
        else:
            print(f"Warning: Test '{args.name}' not found in results")

    # Format JSON with indentation for readability
    formatted_json = json.dumps(summary, indent=2)

    # Output to file or stdout
    if args.output:
        with open(args.output, "w") as f:
            f.write(formatted_json)
        print(f"JSON summary saved to {args.output}")
        print(
            f"To visualize this file, run: uv run python -m mcpx_eval html {args.output}"
        )
    else:
        print(formatted_json)

    # If visualization is requested, create and open it
    output_path = args.html
    html = visualize_json(summary, output_path)
    # Also save a copy to the specified location if provided
    if output_path:
        with open(output_path, "w") as f:
            f.write(html)
        print(f"JSON visualization saved to {output_path}")
        print("To view this visualization again later, open the file in your browser.")
        temp_path = output_path
    if args.show:
        if output_path is None:
            # Write to temporary file and open in browser
            with NamedTemporaryFile(suffix=".html", delete=False, mode="w") as f:
                f.write(html)
                temp_path = f.name

        print("Opening browser...")
        webbrowser.open(f"file://{temp_path}")


async def run():
    from argparse import ArgumentParser

    parser = ArgumentParser("mcpx-eval", description="LLM tool use evaluator")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Main test command (default)
    test_parser = subparsers.add_parser("test", help="Run evaluation tests")
    test_parser.add_argument("--name", default="", help="Test name")
    test_parser.add_argument(
        "--model",
        "-m",
        default=[],
        help="Model to include in test",
        action="append",
    )
    test_parser.add_argument(
        "--tool",
        "-t",
        default=[],
        help="Expected tool",
        action="append",
    )
    test_parser.add_argument(
        "--profile",
        "-p",
        default=None,
        help="Profile to use for all models",
    )

    test_parser.add_argument("--prompt", help="Test prompt")
    test_parser.add_argument("--check", help="Test check")
    test_parser.add_argument(
        "--max-tool-calls", default=None, help="Maximum number of tool calls", type=int
    )
    test_parser.add_argument("--config", help="Test config file")
    test_parser.add_argument(
        "--iter",
        "-i",
        default=1,
        type=int,
        help="Number of times to run the test for each model",
    )

    # Summary command
    summary_parser = subparsers.add_parser("summary", help="Show test results summary")
    summary_parser.add_argument("name", help="Test name to summarize")

    # JSON summary command
    gen_parser = subparsers.add_parser(
        "gen", help="Generate JSON summary of all test data"
    )
    gen_parser.add_argument(
        "--name",
        "-n",
        help="Filter results to a specific test name",
    )
    gen_parser.add_argument(
        "--output",
        "-o",
        help="Output JSON file path (default: print to stdout)",
    )
    gen_parser.add_argument(
        "--show",
        "-s",
        action="store_true",
        help="Create an interactive HTML visualization of the JSON data",
    )
    gen_parser.add_argument(
        "--html",
        help="Output path for HTML visualization (optional)",
    )

    # Global options
    parser.add_argument(
        "--log",
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level.",
    )
    parser.add_argument(
        "--verbose", default=False, action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Setup logging
    level = args.log or "INFO"
    log_level = getattr(logging, level, None)
    if not isinstance(log_level, int):
        raise ValueError("Invalid log level: %s" % level)
    logging.basicConfig(level=log_level)

    if not args.verbose:
        for handler in logging.root.handlers:
            handler.addFilter(logging.Filter("mcpx_eval"))

    # Handle command routing
    command = getattr(args, "command", "test")  # Default to test if not specified

    # Visualization commands removed

    # Summary command
    if command == "summary":
        summary(args)
        return

    # gen command
    elif command == "gen":
        json_summary(args)
        return

    # Test command (default)
    else:
        test = None
        if hasattr(args, "config") and args.config is not None:
            test = Test.load(args.config)
            for model in args.model:
                if args.profile:
                    if "/" in model:
                        a, _ = model.split("/", maxsplit=1)
                        test.models.append(f"{a}/{args.profile}")
                    else:
                        test.models.append(f"{model}/{args.profile}")
                else:
                    test.models.append(model)
            if args.name is None or args.name == "":
                args.name = test.name

        if test is None:
            test = Test(
                name=getattr(args, "name", ""),
                prompt=args.prompt or "",
                check=args.check or "",
                models=args.model,
                max_tool_calls=args.max_tool_calls,
                expected_tools=args.expected_tools,
            )

        iterations = args.iter
        logger.info(
            f"Running {test.name}: {', '.join(test.models)} ({iterations} iteration{'s' if iterations > 1 else ''})"
        )
        judge = Judge(models=test.models)
        judge.db.save_test(test)

        tools = list(judge.agent.client.tools.keys())

        logger.info(f"Found tools: {', '.join(tools)}")

        all_results = []
        total_duration = 0

        for i in range(iterations):
            if iterations > 1:
                logger.info(f"Iteration {i + 1}/{iterations}")

            # For multiple iterations, pass save=True to ensure each run is saved to DB
            res = await judge.run_test(test, save=True)
            total_duration += res.duration
            all_results.extend(res.scores)

            if iterations > 1:
                logger.info(f"Iteration {i + 1} finished in {res.duration}s")

        logger.info(f"{test.name} finished in {total_duration}s total")

        # When multiple iterations are run, only show the last iteration's results
        # to avoid overwhelming the user with output
        results_to_print = all_results if iterations == 1 else res.scores

        if iterations > 1:
            print(f"\nShowing results from iteration {iterations} of {iterations}.")
            print(f"All {iterations} iterations have been saved to the database.")
            print(f"Use 'mcpx-eval summary {test.name}' to see aggregated results.\n")

        for result in results_to_print:
            if result is None:
                continue
            print_result(result)


def main():
    asyncio.run(run())

    # Print helpful usage examples at the end
    print("\nUsage examples:")
    print("  Generate JSON summary:                uv run python -m mcpx_eval gen")
    print(
        "  Open HTML scoreboard in browser:        uv run python -m mcpx_eval gen --show"
    )
    print(
        "  Save JSON to file:                      uv run python -m mcpx_eval gen --json results.json"
    )
    print(
        "  Generate HTML scoreboard:               uv run python -m mcpx_eval gen --html out.html"
    )


if __name__ == "__main__":
    main()
