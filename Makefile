.PHONY: install run debug clean fclean lint lint-strict

install:
	mkdir -p .cache/uv_cache .cache/hf_cache
	UV_CACHE_DIR=.cache/uv_cache \
	HF_HOME=.cache/hf_cache \
	uv sync --python 3.11

run:
# 	uv run python -m src \
# 		--functions_definition data/input/functions_definition.json \
# 		--input data/input/function_calling_tests.json \
# 		--output data/output/function_calls.json

debug:
	uv run python -m pdb -m src $(ARGS)

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .mypy_cache -exec rm -rf {} +
	find . -name .pytest_cache -exec rm -rf {} +
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete

fclean: clean
	rm -rf .cache
	rm -rf .venv

lint:
	uv run flake8 src
	uv run mypy src \
		--warn-return-any \
		--warn-unused-ignores \
		--ignore-missing-imports \
		--disallow-untyped-defs \
		--check-untyped-defs

lint-strict:
	uv run flake8 src
	uv run mypy src --strict