from . import Judge, Test, Database
import asyncio
import logging

logger = logging.getLogger(__name__)


def print_result(result):
    print()
    print(result.model)
    print("=" * len(result.model))
    print()
    print(f"Time: {result.duration}s")
    print("Output:")
    print(result.llm_output)
    print()
    print("Score:")
    print(result.description)
    print("Number of tool calls:", result.tool_calls)
    print("Accuracy:", result.accuracy)
    print("Tool use:", result.tool_use)
    print("Overall:", result.overall)


def summary(args):
    db = Database()
    res = db.average_results(args.name)
    for result in res.scores:
        print_result(result)


async def run():
    from argparse import ArgumentParser

    parser = ArgumentParser("mcpx-eval", description="LLM tool use evaluator")
    parser.add_argument("--name", default="", help="Test name")
    parser.add_argument(
        "--model",
        "-m",
        default=[],
        help="Model to include in test",
        action="append",
    )
    parser.add_argument("--prompt", help="Test prompt")
    parser.add_argument("--check", help="Test check")
    parser.add_argument(
        "--max-tool-calls", default=None, help="Maximum number of tool calls", type=int
    )
    parser.add_argument("--config", help="Test config file")
    parser.add_argument(
        "--log",
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level.",
    )
    parser.add_argument(
        "--verbose", default=False, action="store_true", help="Enable verbose logging"
    )
    parser.add_argument(
        "--db",
        default=False,
        action="store_true",
        help="Summarize results from database",
    )

    args = parser.parse_args()

    level = args.log
    if level is None:
        level = "INFO"

    log_level = getattr(logging, level, None)
    if not isinstance(log_level, int):
        raise ValueError("Invalid log level: %s" % level)
    logging.basicConfig(level=log_level)

    if not args.verbose:
        for handler in logging.root.handlers:
            handler.addFilter(logging.Filter("mcpx_eval"))

    test = None
    if args.config is not None:
        test = Test.load(args.config)
        for model in args.model:
            test.models.append(model)
        if args.name is None or args.name == "":
            args.name = test.name

    if args.db:
        summary(args)
        return

    if test is None:
        test = Test(
            name=args.name,
            prompt=args.prompt,
            check=args.check,
            model=args.model,
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
