.PHONY: setup

# Provision a local virtual environment and install dev dependencies.
setup:
	@if [ ! -d .venv ]; then \
		python -m venv .venv; \
	fi
	@./.venv/bin/python -m pip install --upgrade pip
	@./.venv/bin/python -m pip install -e '.[dev]'
