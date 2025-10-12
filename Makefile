.PHONY: check fmt fmt-black fmt-eslint fmt-check fmt-check-black fmt-check-eslint lint lint-black lint-flake8 lint-eslint type type-mypy type-pyright test test-webui install gitleaks-hook fmt-check-hook hooks web web-build webd webdev devstack-start devstack-stop devstack-status

UV ?= uv
UV_RUN ?= $(UV) run --extra dev
PRECOMMIT_HOOK ?= .git/hooks/pre-commit
WEBUI_NPM ?= npm
GUNICORN ?= gunicorn
GUNICORN_BIND ?= 0.0.0.0:5000
GUNICORN_WORKERS ?= 1
GUNICORN_WORKER_CONNECTIONS ?= 1000
GUNICORN_PID_FILE ?= .gunicorn-web.pid
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

fmt: fmt-black fmt-eslint

fmt-black:
	$(UV_RUN) black .

fmt-eslint:
	cd webui && $(WEBUI_NPM) run format

fmt-check: fmt-check-black fmt-check-eslint

fmt-check-black:
	$(UV_RUN) black --check .

fmt-check-eslint:
	cd webui && $(WEBUI_NPM) run format:check

lint: lint-black lint-flake8 lint-eslint

lint-black:
	$(UV_RUN) black --check .

lint-flake8:
	$(UV_RUN) flake8 .

lint-eslint:
	cd webui && $(WEBUI_NPM) run lint

type: type-mypy type-pyright

type-mypy:
	$(UV_RUN) mypy .

type-pyright:
	$(UV_RUN) pyright

test: type test-webui
	$(UV_RUN) pytest -n auto --cov=llm_judge --cov-report=term-missing --cov-report=json:coverage.json --junitxml=pytest-results.xml

test-webui:
	cd webui && $(WEBUI_NPM) run lint:ci
	cd webui && $(WEBUI_NPM) run test

check: fmt-check lint type test

web-build:
	cd webui && $(WEBUI_NPM) install
	cd webui && $(WEBUI_NPM) run build

web: web-build
	$(GUNICORN) llm_judge.webapp:app --worker-class gevent --workers $(GUNICORN_WORKERS) --worker-connections $(GUNICORN_WORKER_CONNECTIONS) --bind $(GUNICORN_BIND)

webd: web-build
	@PID_FILE=$(GUNICORN_PID_FILE); \
	if [ -f $$PID_FILE ] && kill -0 "$$(cat $$PID_FILE)" 2>/dev/null; then \
		echo "Gunicorn already running with PID $$(cat $$PID_FILE). Stop it with 'kill $$(cat $$PID_FILE)' first."; \
		exit 1; \
	fi; \
		$(GUNICORN) llm_judge.webapp:app --worker-class gevent --workers $(GUNICORN_WORKERS) --worker-connections $(GUNICORN_WORKER_CONNECTIONS) --bind $(GUNICORN_BIND) --daemon --pid $$PID_FILE; \
		while [ ! -f $$PID_FILE ]; do sleep 0.1; done; \
		PID=$$(cat $$PID_FILE); \
		URL=$$(echo "$(GUNICORN_BIND)" | sed 's/^0\.0\.0\.0/127.0.0.1/'); \
		echo "Web dashboard available at http://$$URL/"; \
		echo "Stop it with 'kill $$PID'."

webdev:
	@trap 'kill 0' EXIT INT TERM; \
	PYTHONUNBUFFERED=1 $(UV_RUN) flask --app llm_judge.webapp:create_app run --debug --host 0.0.0.0 --port 5000 & \
	cd webui && $(WEBUI_NPM) install && $(WEBUI_NPM) run dev -- --host 0.0.0.0 --port 5173

devstack-start:
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
