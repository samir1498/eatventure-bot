# EatVenture AI/BOT — `make <target>`
# Fork of JoaoBoll/eatventure-bot (cloned as eatventure-bot).
# Vision: H.264 scrcpy-server stream + two-stage template matching + state machine.
PROJ := /home/samir/Developer/personal/eatventure-24-7/eatventure-bot
VENV := $(PROJ)/.venv
PY := $(VENV)/bin/python
DEVICE ?= emulator-5554

.PHONY: help install run test check clean

help: ## show this help
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*##"} {printf "  %-10s %s\n", $$1, $$2}'

install: ## create venv + install dependencies
	python3 -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -r $(PROJ)/requirements.txt

run: ## start the bot (uses DEVICE env or default emulator-5554)
	cd $(PROJ) && $(PY) src/main.py --device $(DEVICE)

test: ## run all tests
	cd $(PROJ) && $(PY) tests/test_detection.py && $(PY) tests/test_state_machine.py && $(PY) tests/test_pipeline.py

check: ## quick health check (device + capture)
	cd $(PROJ) && $(PY) -c "from src.core.config import DEVICE_SERIAL; print('device:', DEVICE_SERIAL)"

clean: ## remove venv + caches
	rm -rf $(VENV) $(PROJ)/__pycache__ $(PROJ)/src/__pycache__ $(PROJ)/src/*/__pycache__