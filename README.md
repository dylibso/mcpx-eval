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

Or using `uvx` without installation:

```bash
uvx mcpx-eval
```

## mcp.run Setup

You will need to get an mcp.run session ID by running:

```bash
npx --yes -p @dylibso/mcpx gen-session --write
```

This will generate a new session and write the session ID to a configuration file that can be used
by `mcpx-eval`.
 
If you need to store the session ID in  an environment variable you can run `gen-session`
without the `--write` flag:

```bash
npx --yes -p @dylibso/mcpx gen-session
```

which should output something like:

```
Login successful!
Session: kabA7w6qH58H7kKOQ5su4v3bX_CeFn4k.Y4l/s/9dQwkjv9r8t/xZFjsn2fkLzf+tkve89P1vKhQ
```

Then set the `MCP_RUN_SESSION_ID` environment variable:

```
$ export MCP_RUN_SESSION_ID=kabA7w6qH58H7kKOQ5su4v3bX_CeFn4k.Y4l/s/9dQwkjv9r8t/xZFjsn2fkLzf+tkve89P1vKhQ
```

## Usage

Run an eval comparing all mcp.task runs for `my-task`:

```bash
mcpx-eval test --task my-task --task-run all
```

Only evaluate the latest task run:

```bash
mcpx-eval test --task my-task --task-run latest
```

Or trigger a new task run:

```bash
mcpx-eval test --task my-task --task-run new
```

Run an mcp.run task locally with a different set of models:

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
- `task-run` - optional, one of `latest`, `new`, `all` or the name/index of the task run to analyze
- `prompt` - prompt to test, this is passed to the LLM under test, this can be left blank if `task` is set
- `check` - prompt for the judge, this is used to determine the quality of the test output 
- `expected-tools` - list of tool names that might be used
- `ignored-tools` - optional, list of tools to ignore, they will not be available to the LLM
- `import` - optional, includes fields from another test TOML file
- `vars` - optional, a dict of variables that will be used to format the prompt
