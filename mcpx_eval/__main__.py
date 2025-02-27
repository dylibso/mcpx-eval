from . import Judge, Test, Database
from .html import visualize_json
import asyncio
import logging
import json
import os
from datetime import datetime
from tempfile import NamedTemporaryFile
import webbrowser

logger = logging.getLogger(__name__)


def print_result(result):
    print()
    print(f"{result.model}")
    print("=" * len(result.model))
    print()
    print(f"Time: {result.duration}s")
    print("Output:")
    print(result.llm_output)
    print()
    print("Score:")
    print(result.description)

    print("Number of tool calls:", result.tool_calls)
    if result.redundant_tool_calls > 0:
        print("Redundant tool calls:", result.redundant_tool_calls)
    print("Tool use:", result.tool_use)
    print("Accuracy:", result.accuracy)
    print("Clarity:", result.clarity)
    print("Helpfulness:", result.helpfulness)

    # Hallucination metrics
    print("Hallucination score:", result.hallucination_score)
    if result.false_claims and len(result.false_claims) > 0:
        print("False claims detected:")
        for claim in result.false_claims:
            print(f"  - {claim}")

    # Tool analysis
    if result.tool_analysis and len(result.tool_analysis) > 0:
        print("\nTool analysis:")
        for tool_id, analysis in result.tool_analysis.items():
            if isinstance(analysis, list):
                for a in analysis:
                    print(f"  {tool_id}: {a['name']}")
                    print(f"    - Redundancy: {a['redundancy']}")
            else:
                print(f"  {tool_id}: {analysis['name']}")
                print(f"    - Redundancy: {analysis['redundancy']}")

    print("\nOverall score:", result.overall)


def summary(args):
    db = Database()
    res = db.average_results(args.name)
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
    output_path = args.viz_output
    html = visualize_json(summary, output_path)
    # Also save a copy to the specified location if provided
    if output_path:
        with open(output_path, "w") as f:
            f.write(html)
        print(f"JSON visualization saved to {output_path}")
        print(f"To view this visualization again later, open the file in your browser.")
        temp_path = output_path
    if args.show:
        if output_path is None:
            # Write to temporary file and open in browser
            with NamedTemporaryFile(suffix=".html", delete=False, mode="w") as f:
                f.write(html)
                temp_path = f.name

        print(f"Opening browser...")
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
    test_parser.add_argument("--prompt", help="Test prompt")
    test_parser.add_argument("--check", help="Test check")
    test_parser.add_argument(
        "--max-tool-calls", default=None, help="Maximum number of tool calls", type=int
    )
    test_parser.add_argument("--config", help="Test config file")
    test_parser.add_argument(
        "--iter",
        default=1,
        type=int,
        help="Number of times to run the test for each model",
    )

    # Summary command
    summary_parser = subparsers.add_parser("summary", help="Show test results summary")
    summary_parser.add_argument("name", help="Test name to summarize")

    # JSON summary command
    json_parser = subparsers.add_parser(
        "json", help="Generate JSON summary of all test data"
    )
    json_parser.add_argument(
        "--name",
        "-n",
        help="Filter results to a specific test name",
    )
    json_parser.add_argument(
        "--output",
        "-o",
        help="Output JSON file path (default: print to stdout)",
    )
    json_parser.add_argument(
        "--show",
        "-s",
        action="store_true",
        help="Create an interactive HTML visualization of the JSON data",
    )
    json_parser.add_argument(
        "--viz-output",
        help="Output path for HTML visualization (optional)",
    )

    # JSON visualization command (standalone)
    viz_json_parser = subparsers.add_parser(
        "html", help="Visualize JSON data from a file"
    )
    viz_json_parser.add_argument(
        "input",
        help="Input JSON file path",
    )
    viz_json_parser.add_argument(
        "--output",
        "-o",
        help="Output HTML file path (optional)",
    )

    # Visualization commands removed

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

    # JSON summary command
    elif command == "json":
        json_summary(args)
        return

    # JSON visualization command
    elif command == "html":
        import json

        try:
            with open(args.input, "r") as f:
                data = json.load(f)
            visualize_json(data, args.output)
        except FileNotFoundError:
            print(f"Error: File '{args.input}' not found.")
            print(
                f"Generate a JSON file first with: uv run python -m mcpx_eval json -o {args.input}"
            )
            return
        except json.JSONDecodeError:
            print(f"Error: '{args.input}' is not a valid JSON file.")
            return

    # Test command (default)
    else:
        test = None
        if hasattr(args, "config") and args.config is not None:
            test = Test.load(args.config)
            for model in args.model:
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
            )

        iterations = args.iter
        logger.info(
            f"Running {test.name}: {', '.join(test.models)} ({iterations} iteration{'s' if iterations > 1 else ''})"
        )
        judge = Judge(models=test.models)
        judge.db.save_test(test)

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
    print("  Generate JSON summary:                uv run python -m mcpx_eval json")
    print(
        "  Generate and visualize JSON:          uv run python -m mcpx_eval json --show"
    )
    print(
        "  Save JSON to file:                    uv run python -m mcpx_eval json -o results.json"
    )
    print(
        "  Visualize existing JSON file:         uv run python -m mcpx_eval html results.json"
    )
    print(
        "  Save visualization to HTML:           uv run python -m mcpx_eval html results.json -o viz.html"
    )


if __name__ == "__main__":
    main()
