PYTHON ?= /usr/local/bin/python3
STREAMLIT ?= streamlit

.PHONY: install-dev test test-ingestion test-retrieval test-generation run-ui

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

run-ui:
	$(STREAMLIT) run app/streamlit_app.py