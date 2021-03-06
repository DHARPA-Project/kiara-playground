.PHONY: clean clean-test clean-pyc clean-build docs help
.DEFAULT_GOAL := help

help:
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

docs: ## build documentation
	mkdocs build

serve-docs: ## serve and watch documentation
	mkdocs serve -a 0.0.0.0:8000

clean: clean-build clean-pyc clean-test ## remove all build, test, coverage and Python artifacts

clean-build: ## remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg' -exec rm -f {} +

clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test: ## remove test and coverage artifacts
	rm -fr .tox/
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr .pytest_cache
	rm -fr .mypy_cache

init: clean ## initialize a development environment (to be run in virtualenv)
	git init
	git checkout -b develop || true
	pip install pip==21.2.4 setuptools==57.4.0
	pip install --extra-index-url https://pypi.fury.io/dharpa/ -U -e '.[all_dev]'
	pip install  --extra-index-url https://pypi.fury.io/dharpa/ -U 'kiara_modules.core[all]' 'kiara_modules.language_processing[all]' 'kiara_modules.language_processing[all]' 'kiara_modules.network_analysis[all]' 'kiara.streamlit[all]'
#	pre-commit install
#	pre-commit install --hook-type commit-msg
	setup-cfg-fmt setup.cfg || true
	git add "*" ".*"
	#pre-commit run --all-files || true
	git add "*" ".*"

update:
	git pull
	pip install pip==21.2.4 setuptools==57.4.0
	pip install --extra-index-url https://pypi.fury.io/dharpa/ -e '.[all_dev]'
	pip install --extra-index-url https://pypi.fury.io/dharpa/ -U 'kiara[all]' 'kiara_modules.core[all]' 'kiara_modules.language_processing[all]' 'kiara_modules.network_analysis[all]' 'kiara.streamlit[all]'

update-dependencies:  ## update all development dependencies
	pip install pip==21.2.4 setuptools==57.4.0
	pip install --extra-index-url https://pypi.fury.io/dharpa/ -e '.[all_dev]'
	pip install --extra-index-url https://pypi.fury.io/dharpa/ -U 'kiara[all]' 'kiara_modules.core[all]' 'kiara_modules.language_processing[all]' 'kiara_modules.network_analysis[all]' 'kiara.streamlit[all]'

update-dependencies-dev:  ## update all development dependencies
	pip install pip==21.2.4 setuptools==57.4.0
	pip install --extra-index-url https://pypi.fury.io/dharpa/ -e '.[all_dev]'
	pip install --extra-index-url https://pypi.fury.io/dharpa/ -U 'git+https://github.com/DHARPA-Project/kiara.git@develop#egg=kiara[all]' 'git+https://github.com/DHARPA-Project/kiara_modules.core.git@develop#egg=kiara_modules.core[all]' 'git+https://github.com/DHARPA-Project/kiara_modules.language_processing.git@develop#egg=kiara_modules.language_processing[all]' 'git+https://github.com/DHARPA-Project/kiara_modules.network_analysis.git@develop#egg=kiara_modules.network_analysis[all]' 'git+https://github.com/frkl-io/kiara.streamlit.git@develop#egg=kiara.streamlit[all]'

setup-cfg-fmt: # format setup.cfg
	setup-cfg-fmt setup.cfg || true

black: ## run black
	black --config pyproject.toml setup.py src/kiara_modules/playground tests

flake: ## check style with flake8
	flake8 src/kiara_modules/playground tests

mypy: ## run mypy
	mypy  --namespace-packages --explicit-package-base src/kiara_modules/playground

test: ## run tests quickly with the default Python
	py.test

test-all: ## run tests on every Python version with tox
	tox

coverage: ## check code coverage quickly with the default Python
	coverage run -m pytest tests
	coverage report -m

check: black flake mypy test ## run dev-related checks

pre-commit: ## run pre-commit on all files
	pre-commit run --all-files

dist: clean ## build source and wheel packages
	python setup.py sdist
	python setup.py bdist_wheel
	ls -l dist
