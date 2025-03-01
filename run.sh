#!/usr/bin/env bash

remote_models="\
--model o1 \
--model gpt-4o \
--model claude-3-5-sonnet-latest \
--model claude-3-7-sonnet-latest
--model claude-3-5-haiku-latest"

models=${models-$remote_models}
iter=${iterations-5}

for test in tests/*.toml; do
  uv run mcpx-eval test --config $test $models --iter $iter 
done
