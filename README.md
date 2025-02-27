# mcpx-eval

A tool for evaluating open-ended tool use across various LLMs

## Usage

Run a test with 10 iterations:

```bash
uv run mcpx-eval test --model ... --model ... --config tests/factorial-5.toml --iter 10
```

Summarize all tests as an HTML table:

```bash
uv run mcpx-eval html -o eval.html
```
