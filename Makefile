SHELL:=/bin/bash
ROOT_DIR = $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PY_FILES = $(shell find $(ROOT_DIR) | grep -v .eggs | grep -v ^$(ROOT_DIR)/env | grep \.py$$ 2>/dev/null)
all: dependencies

venv:
	if [ ! -d $(ROOT_DIR)/venv ]; then python3 -m venv $(ROOT_DIR)/venv; fi

dependencies: venv
	source $(ROOT_DIR)/venv/bin/activate && yes w | python -m pip install --upgrade pip
	source $(ROOT_DIR)/venv/bin/activate && yes w | python -m pip install --upgrade --progress-bar off --upgrade-strategy eager -e .
	source $(ROOT_DIR)/venv/bin/activate && yes w | python -m pip install --upgrade --progress-bar off --upgrade-strategy eager -e fast-transformers/.
clean:
	rm -Rf $(ROOT_DIR)/venv