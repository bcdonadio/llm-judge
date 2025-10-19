.PHONY: check fmt format-python format-webui format-check format-check-python format-check-webui lint lint-python lint-webui lint-webui-report typing typing-mypy typing-pyright typing-webui type unit-tests unit-tests-python unit-tests-webui test test-webui install gitleaks-hook format-check-hook hooks web web-build webd webdev devstack-start devstack-stop devstack-status deadcode deadcode-python deadcode-webui

UV ?= uv
UV_RUN ?= $(UV) run --extra dev
PRECOMMIT_HOOK ?= .git/hooks/pre-commit
WEBUI_NPM ?= npm
UVICORN ?= uvicorn
UVICORN_HOST ?= 0.0.0.0
UVICORN_PORT ?= 5000
UVICORN_WORKERS ?= 1
UVICORN_PID_FILE ?= .uvicorn-web.pid
DEVSTACK ?= $(UV_RUN) python -m llm_judge.devstack
DEVSTACK_BACKEND_HOST ?= 127.0.0.1
DEVSTACK_BACKEND_PORT ?= 5000
DEVSTACK_FRONTEND_HOST ?= 127.0.0.1
DEVSTACK_FRONTEND_PORT ?= 5173
DEVSTACK_LOG_DIR ?= .devstack

install: hooks
	$(UV) sync --extra dev
	@if [ -n "$$CI" ]; then \
		cd webui && $(WEBUI_NPM) ci; \
	else \
		cd webui && $(WEBUI_NPM) install; \
	fi

hooks: gitleaks-hook format-check-hook

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

format-check-hook:
	@mkdir -p $(dir $(PRECOMMIT_HOOK))
	@if [ ! -f $(PRECOMMIT_HOOK) ]; then \
		echo '#!/usr/bin/env bash' > $(PRECOMMIT_HOOK); \
		echo 'set -e' >> $(PRECOMMIT_HOOK); \
	fi
	@if ! grep -Fq '# format-check pre-commit hook' $(PRECOMMIT_HOOK); then \
		printf '\n# format-check pre-commit hook\n' >> $(PRECOMMIT_HOOK); \
		echo 'if ! make format-check; then' >> $(PRECOMMIT_HOOK); \
		echo '  echo "Formatting check failed. Run '\''make fmt'\'' to fix formatting."' >> $(PRECOMMIT_HOOK); \
		echo '  exit 1' >> $(PRECOMMIT_HOOK); \
		echo 'fi' >> $(PRECOMMIT_HOOK); \
	fi
	@chmod +x $(PRECOMMIT_HOOK)

fmt: format-python format-webui

format-python:
	$(UV_RUN) black .

format-webui:
	cd webui && $(WEBUI_NPM) run format

format-check: format-check-python format-check-webui

format-check-python:
	$(UV_RUN) black --check .

format-check-webui:
	cd webui && $(WEBUI_NPM) run format-check

fmt-check: format-check

lint: lint-python lint-webui

lint-python:
	$(UV_RUN) flake8 .

lint-webui:
	cd webui && $(WEBUI_NPM) run lint

lint-webui-report:
	cd webui && $(WEBUI_NPM) run lint:report

deadcode: deadcode-python deadcode-webui

deadcode-python:
	$(UV_RUN) vulture

deadcode-webui:
	cd webui && $(WEBUI_NPM) exec knip

typing: typing-mypy typing-pyright typing-webui

typing-mypy:
	$(UV_RUN) mypy .

typing-pyright:
	$(UV_RUN) pyright

typing-webui:
	cd webui && $(WEBUI_NPM) run typing

type: typing

unit-tests: unit-tests-python unit-tests-webui

unit-tests-python:
	$(UV_RUN) pytest -n auto --cov=llm_judge --cov-report=term-missing --cov-report=json:coverage.json --junitxml=pytest-results.xml

unit-tests-webui:
	cd webui && $(WEBUI_NPM) run unit-test

test: typing unit-tests

test-webui: lint-webui-report unit-tests-webui

check: format-check lint typing unit-tests deadcode

web-build:
	cd webui && $(WEBUI_NPM) install
	cd webui && $(WEBUI_NPM) run build

web: web-build
	$(UV_RUN) $(UVICORN) llm_judge.webapp:app --host $(UVICORN_HOST) --port $(UVICORN_PORT) --workers $(UVICORN_WORKERS)

webd: web-build
	@PID_FILE=$(UVICORN_PID_FILE); \
	if [ -f $$PID_FILE ] && kill -0 "$$(cat $$PID_FILE)" 2>/dev/null; then \
		echo "Uvicorn already running with PID $$(cat $$PID_FILE). Stop it with 'kill $$(cat $$PID_FILE)' first."; \
		exit 1; \
	fi; \
		nohup $(UV_RUN) $(UVICORN) llm_judge.webapp:app --host $(UVICORN_HOST) --port $(UVICORN_PORT) --workers $(UVICORN_WORKERS) > /dev/null 2>&1 & \
		echo $$! > $$PID_FILE; \
		sleep 1; \
		PID=$$(cat $$PID_FILE); \
		URL=$$(echo "$(UVICORN_HOST):$(UVICORN_PORT)" | sed 's/^0\.0\.0\.0/127.0.0.1/'); \
		echo "Web dashboard available at http://$$URL/"; \
		echo "Stop it with 'kill $$PID'."

webdev:
	@trap 'kill 0' EXIT INT TERM; \
	PYTHONUNBUFFERED=1 $(UV_RUN) $(UVICORN) llm_judge.webapp:app --host 0.0.0.0 --port 5000 --reload & \
	cd webui && $(WEBUI_NPM) install && $(WEBUI_NPM) run dev -- --host 0.0.0.0 --port 5173

devstack-start: web-build
	$(DEVSTACK) start \
		--backend-host $(DEVSTACK_BACKEND_HOST) \
		--backend-port $(DEVSTACK_BACKEND_PORT) \
		--frontend-host $(DEVSTACK_FRONTEND_HOST) \
		--frontend-port $(DEVSTACK_FRONTEND_PORT) \
		--log-dir $(DEVSTACK_LOG_DIR)

devstack-stop:
	$(DEVSTACK) stop $(if $(FORCE),--force,)

devstack-status:
	$(DEVSTACK) status
