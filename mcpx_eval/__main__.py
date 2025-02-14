from . import Judge, Test
import asyncio
import logging

logger = logging.getLogger(__name__)


async def run():
    from argparse import ArgumentParser

    parser = ArgumentParser("mcpx-eval", description="LLM tool use evaluator")
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
    parser.add_argument("tests", help="Test files", nargs="*")
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

    tests = [Test.load(t) for t in args.tests]

    if len(args.model) > 0:
        tests.append(
            Test(
                "command-line",
                args.prompt,
                args.check,
                args.model,
                max_tool_calls=args.max_tool_calls,
            )
        )


    for test in tests:
        logger.info(f"Running {test.name}: {', '.join(test.models)}")
        judge = Judge(models=test.models)
        res = await judge.run_test(test)
        logger.info(f"{test.name} finished in {res.duration}")
        for result in res.scores:
            print()
            print(result.model)
            print("=" * len(result.model))
            print()
            print(f"Time: {result.duration}s")
            print("Output:")
            print(result.output)
            print()
            print("Score:")
            print(result.description)
            print("Number of tool calls:", result.tool_calls)
            print("Accuracy:", result.accuracy)
            print("Tool use:", result.tool_use)
            print("Overall:", result.overall)


def main():
    asyncio.run(run())


if __name__ == "__main__":
    main()
