.PHONY: check fmt fmt-check lint lint-black lint-flake8 type type-mypy type-pyright test install gitleaks-hook fmt-check-hook hooks web web-build

UV ?= uv
UV_RUN ?= $(UV) run --extra dev
PRECOMMIT_HOOK ?= .git/hooks/pre-commit
WEBUI_NPM ?= npm
GUNICORN ?= gunicorn
GUNICORN_BIND ?= 0.0.0.0:5000
GUNICORN_WORKERS ?= 1
GUNICORN_WORKER_CONNECTIONS ?= 1000

install: hooks
	$(UV) sync --extra dev

hooks: gitleaks-hook fmt-check-hook

gitleaks-hook:
	@mkdir -p $(dir $(PRECOMMIT_HOOK))
	@if [ ! -f $(PRECOMMIT_HOOK) ]; then \
		echo '#!/usr/bin/env bash' > $(PRECOMMIT_HOOK); \
		echo 'set -e' >> $(PRECOMMIT_HOOK); \
	fi
	@if ! grep -Fq '# gitleaks pre-commit hook' $(PRECOMMIT_HOOK); then \
		printf '\n# gitleaks pre-commit hook\n' >> $(PRECOMMIT_HOOK); \
		echo 'if command -v gitleaks >/dev/null 2>&1; then' >> $(PRECOMMIT_HOOK); \
		echo '  gitleaks protect --staged --no-banner --redact' >> $(PRECOMMIT_HOOK); \
		echo 'else' >> $(PRECOMMIT_HOOK); \
		echo '  echo "gitleaks not found; install it to run this hook."' >> $(PRECOMMIT_HOOK); \
		echo '  exit 1' >> $(PRECOMMIT_HOOK); \
		echo 'fi' >> $(PRECOMMIT_HOOK); \
	fi
	@chmod +x $(PRECOMMIT_HOOK)

fmt-check-hook:
	@mkdir -p $(dir $(PRECOMMIT_HOOK))
	@if [ ! -f $(PRECOMMIT_HOOK) ]; then \
		echo '#!/usr/bin/env bash' > $(PRECOMMIT_HOOK); \
		echo 'set -e' >> $(PRECOMMIT_HOOK); \
	fi
	@if ! grep -Fq '# fmt-check pre-commit hook' $(PRECOMMIT_HOOK); then \
		printf '\n# fmt-check pre-commit hook\n' >> $(PRECOMMIT_HOOK); \
		echo 'if ! make fmt-check; then' >> $(PRECOMMIT_HOOK); \
		echo '  echo "Formatting check failed. Run '\''make fmt'\'' to fix formatting."' >> $(PRECOMMIT_HOOK); \
		echo '  exit 1' >> $(PRECOMMIT_HOOK); \
		echo 'fi' >> $(PRECOMMIT_HOOK); \
	fi
	@chmod +x $(PRECOMMIT_HOOK)

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

web-build:
	cd webui && $(WEBUI_NPM) install
	cd webui && $(WEBUI_NPM) run build

web: web-build
	$(GUNICORN) llm_judge.webapp:app --worker-class gevent --workers $(GUNICORN_WORKERS) --worker-connections $(GUNICORN_WORKER_CONNECTIONS) --bind $(GUNICORN_BIND)
