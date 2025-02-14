from . import Judge, Test

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
    parser.add_argument("tests", help="Test files", nargs="*")

    args = parser.parse_args()

    tests = [Test.load(t) for t in args.tests]

    if len(args.model) > 0:
        tests.append(Test("command-line", args.prompt, args.check, args.model))

    for test in tests:
        print(f"Running {test.name}: {', '.join(test.models)}")
        judge = Judge(models=test.models)
        res = await judge.run(test.prompt, test.check)
        for result in res.scores:
            print()
            print(result.model)
            print("=" * len(result.model))
            print()
            print("Output:")
            print(result.output)
            print()
            print("Score:")
            print(result.description)
            print("Accuracy:", result.accuracy)
            print("Tool use:", result.tool_use)
            print("Overall:", result.overall)


def main():
    import asyncio

    asyncio.run(run())

if __name__ == "__main__":
    main()
