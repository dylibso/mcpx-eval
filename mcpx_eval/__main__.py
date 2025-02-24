from . import Judge, Test, Database
import asyncio
import logging
import json
import os
from datetime import datetime

logger = logging.getLogger(__name__)


def print_result(result):
    print()
    print(f"{result.model} ({result.model_version})")
    print("=" * (len(result.model) + len(result.model_version) + 3))
    print()
    print(f"Time: {result.duration}s")
    print("Output:")
    print(result.llm_output)
    print()
    print("Score:")
    print(result.description)
    
    print("Performance metrics:")
    print("Number of tool calls:", result.tool_calls)
    print("Accuracy:", result.accuracy)
    print("Tool use:", result.tool_use)
    
    print("\nUser-perceived quality:")
    print("Clarity:", result.clarity)
    print("Helpfulness:", result.helpfulness)
    print("User alignment:", result.user_aligned)
    
    print("\nOverall score:", result.overall)


def summary(args):
    db = Database()
    res = db.average_results(args.name)
    for result in res.scores:
        print_result(result)
        
def create_visualization(args):
    db = Database()
    
    # Parse test names from comma-separated string
    test_names = [name.strip() for name in args.tests.split(",")]
    
    # Get data for each test
    test_data = {}
    for test_name in test_names:
        results = db.average_results(test_name)
        test_data[test_name] = results
    
    # Create visualization data based on chart type
    chart_data = {}
    
    if args.type == "bar":
        # Create bar chart comparing models across tests
        models = set()
        metrics = ["accuracy", "tool_use", "clarity", "helpfulness", "user_aligned", "overall"]
        
        for test_name, results in test_data.items():
            for score in results.scores:
                models.add(f"{score.model}:{score.model_version}")
        
        for metric in metrics:
            chart_data[metric] = {model: {} for model in models}
            
            for test_name, results in test_data.items():
                for score in results.scores:
                    model_key = f"{score.model}:{score.model_version}"
                    chart_data[metric][model_key][test_name] = getattr(score, metric)
    
    elif args.type == "radar":
        # Create radar chart comparing metrics across models for each test
        for test_name, results in test_data.items():
            chart_data[test_name] = {}
            for score in results.scores:
                model_key = f"{score.model}:{score.model_version}"
                chart_data[test_name][model_key] = {
                    "accuracy": score.accuracy,
                    "tool_use": score.tool_use,
                    "clarity": score.clarity,
                    "helpfulness": score.helpfulness,
                    "user_aligned": score.user_aligned,
                    "overall": score.overall
                }
    
    # Save visualization
    db.create_visualization(
        name=args.name,
        description=args.description,
        test_names=test_names,
        chart_type=args.type,
        chart_data=chart_data
    )
    
    print(f"Visualization '{args.name}' created successfully!")

def list_visualizations(args):
    db = Database()
    visualizations = db.get_visualizations()
    
    if not visualizations:
        print("No visualizations found.")
        return
        
    print("\nAvailable visualizations:")
    print("=" * 50)
    
    for viz in visualizations:
        viz_id, name, description, test_names, chart_type = viz
        tests = json.loads(test_names)
        
        print(f"ID: {viz_id}")
        print(f"Name: {name}")
        print(f"Type: {chart_type}")
        print(f"Tests: {', '.join(tests)}")
        print(f"Description: {description}")
        print("-" * 50)


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
    
    # Summary command
    summary_parser = subparsers.add_parser("summary", help="Show test results summary")
    summary_parser.add_argument("name", help="Test name to summarize")
    
    # Visualization commands
    viz_parser = subparsers.add_parser("viz", help="Visualization subcommands")
    viz_subparsers = viz_parser.add_subparsers(dest="viz_command", help="Visualization command")
    
    # Create visualization
    create_viz_parser = viz_subparsers.add_parser("create", help="Create a new visualization")
    create_viz_parser.add_argument("name", help="Name for the visualization")
    create_viz_parser.add_argument("--description", "-d", default="", help="Description of the visualization")
    create_viz_parser.add_argument("--tests", "-t", required=True, help="Comma-separated list of test names to include")
    create_viz_parser.add_argument("--type", choices=["bar", "radar"], default="bar", help="Type of visualization")
    
    # List visualizations
    list_viz_parser = viz_subparsers.add_parser("list", help="List saved visualizations")
    
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
    
    # Visualization commands
    if command == "viz":
        viz_command = getattr(args, "viz_command", None)
        if viz_command == "create":
            create_visualization(args)
            return
        elif viz_command == "list":
            list_visualizations(args)
            return
        else:
            parser.print_help()
            return
    
    # Summary command
    elif command == "summary":
        summary(args)
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
                name=args.name,
                prompt=args.prompt or "",
                check=args.check or "",
                models=args.model,
                max_tool_calls=args.max_tool_calls,
            )

        logger.info(f"Running {test.name}: {', '.join(test.models)}")
        judge = Judge(models=test.models)
        judge.db.save_test(test)
        res = await judge.run_test(test)
        logger.info(f"{test.name} finished in {res.duration}")
        
        for result in res.scores:
            if result is None:
                continue
            print_result(result)


def main():
    asyncio.run(run())


if __name__ == "__main__":
    main()
