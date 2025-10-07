.PHONY: check fmt lint lint-black lint-flake8 type type-mypy type-pyright test install

UV ?= uv
UV_RUN ?= $(UV) run --extra dev

install:
	$(UV) sync --extra dev

fmt:
	$(UV_RUN) black .

lint: lint-black lint-flake8

lint-black:
	$(UV_RUN) black --check .

lint-flake8:
	$(UV_RUN) flake8 .

type: type-mypy type-pyright

type-mypy:
	$(UV_RUN) mypy .

type-pyright:
	$(UV_RUN) pyright

test: type
	$(UV_RUN) pytest -n auto --cov=llm_judge --cov-report=term-missing

check: lint type test
