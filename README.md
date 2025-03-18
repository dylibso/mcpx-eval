# mcpx-eval

A framework for evaluating open-ended tool use across various large language models.

`mcpx-eval` can be used to compare the output of different LLMs with the same prompt for a given task using [mcp.run](https://www.mcp.run) tools.
This means we're not only interested in the quality of the output, but also curious about the helpfulness of various models
when presented with real world tools.

## Test configs

The [tests/](https://github.com/dylibso/mcpx-eval/tree/main/tests) directory contains pre-defined evals

## Installation


```bash
uv tool install mcpx-eval
```


Or from git:

```bash
uv tool install git+https://github.com/dylibso/mcpx-eval
```

## Usage

Run the `my-test` test for 10 iterations:

```bash
mcpx-eval test --model ... --model ... --config my-test.toml --iter 10
```

Or run a task directly from mcp.run:

```bash
mcpx-eval test --model .. --model .. --task my-task --iter 10
```

Generate an HTML scoreboard for all evals:

```bash
mcpx-eval gen --html results.html --show
```

### Test file

A test file is a TOML file containing the following fields:

- `name` - name of the test
- `task` - optional, the name of the mcp.run task to use
- `prompt` - prompt to test, this is passed to the LLM under test, this can be left blank if `task` is set
- `check` - prompt for the judge, this is used to determine the quality of the test output 
- `expected-tools` - list of tool names that might be used
- `ignore-tools` - optional, list of tools to ignore, they will not be available to the LLM
- `import` - optional, includes fields from another test TOML file
- `vars` - optional, a dict of variables that will be used to format the prompt
