.PHONY: check fmt fmt-check lint lint-black lint-flake8 type type-mypy type-pyright test install gitleaks-hook

UV ?= uv
UV_RUN ?= $(UV) run --extra dev
GITLEAKS_HOOK ?= .git/hooks/pre-commit

install: gitleaks-hook
	$(UV) sync --extra dev

gitleaks-hook:
	@mkdir -p $(dir $(GITLEAKS_HOOK))
	@if [ ! -f $(GITLEAKS_HOOK) ]; then \
		echo '#!/usr/bin/env bash' > $(GITLEAKS_HOOK); \
		echo 'set -e' >> $(GITLEAKS_HOOK); \
	fi
	@if ! grep -Fq '# gitleaks pre-commit hook' $(GITLEAKS_HOOK); then \
		printf '\n# gitleaks pre-commit hook\n' >> $(GITLEAKS_HOOK); \
		echo 'if command -v gitleaks >/dev/null 2>&1; then' >> $(GITLEAKS_HOOK); \
		echo '  gitleaks protect --staged --no-banner --redact' >> $(GITLEAKS_HOOK); \
		echo 'else' >> $(GITLEAKS_HOOK); \
		echo '  echo "gitleaks not found; install it to run this hook."' >> $(GITLEAKS_HOOK); \
		echo '  exit 1' >> $(GITLEAKS_HOOK); \
		echo 'fi' >> $(GITLEAKS_HOOK); \
	fi
	@chmod +x $(GITLEAKS_HOOK)

fmt:
	$(UV_RUN) black .

fmt-check:
	$(UV_RUN) black --check .

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

check: fmt-check lint type test
