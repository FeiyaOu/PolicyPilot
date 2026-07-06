PYTHON ?= /usr/local/bin/python3

.PHONY: install-dev test test-ingestion test-retrieval test-generation

install-dev:
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -r requirements-dev.txt

test:
	$(PYTHON) -m pytest

test-ingestion:
	$(PYTHON) -m pytest tests/ingestion

test-retrieval:
	$(PYTHON) -m pytest tests/retrieval

test-generation:
	$(PYTHON) -m pytest tests/generation