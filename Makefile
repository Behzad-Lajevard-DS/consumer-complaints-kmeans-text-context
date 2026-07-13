PYTHON ?= python

.PHONY: install notebook validate

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

notebook:
	jupyter lab notebooks/consumer_complaints_kmeans.ipynb

validate:
	$(PYTHON) scripts/validate_repository.py
