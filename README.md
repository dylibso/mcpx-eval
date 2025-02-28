# mcpx-eval

A tool for evaluating open-ended tool use across various LLMs

## Usage

Run the `factorial-5` test with 10 iterations:

```bash
uv run mcpx-eval test --model ... --model ... --config tests/factorial-5.toml --iter 10
```

Summarize all tests as an HTML table and open a browser:

```bash
uv run mcpx-eval gen --html results.html --show
```
