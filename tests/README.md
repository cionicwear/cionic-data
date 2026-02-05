# Tests

The tests directory contains unit and integration tests for the cionic data tools.

## Setup

The tests depend on third party python packages including numpy, pandas, scipy, matplotlib, requests, and development tools (black, flake8, isort, pre_commit, pytest).

These packages can be installed into your environment with the following commands

Use a Virtual Environment in the project root:

`python3 -m venv venv`

Activate the virtual environment:

`source venv/bin/activate`

Install packages:

`pip3 install -r jupyter/requirements.txt`

Set up pre-commit hooks:

`pre-commit install`

## Running Tests

Run all tests:

`pytest`

Run tests with verbose output:

`pytest -v`

Run a specific test file:

`pytest tests/test_api.py`

Run a specific test class:

`pytest tests/test_api.py::TestIncludeEulersToNpz`

Run a specific test function:

`pytest tests/test_api.py::TestIncludeEulersToNpz::test_include_eulers_basic`

Run tests with coverage:

`pytest --cov=cionic --cov-report=html`

## Test Files

- `test_api.py`: Tests for API functions that add computed data to NPZ files
- `test_download.py`: Tests for download functionality
- `test_foot_segmenter.py`: Tests for foot segmentation algorithms
- `test_gait_metrics.py`: Tests for gait metrics calculations
- `test_metrics_calculator.py`: Tests for metrics calculator
- `test_npz_utils.py`: Tests for NPZ utility functions
- `test_pod_utils.py`: Tests for POD utility functions
- `test_runnable_nbs.py`: Tests for runnable notebooks
- `test_segmenter.py`: Tests for segmentation functionality
- `test_stats.py`: Tests for statistical functions

## Test Fixtures

Test fixtures are located in the `fixtures/` directory. See [fixtures/README.md](fixtures/README.md) for more information about test data and how to regenerate fixtures.

## Committing changes with pre-commit hooks

Pushing changes requires passing formatting and linting standards integrated into pre-commit hooks. These will automatically run when you try to commit, and the commit will be blocked if formatting or linting checks fail. It is convenient to check if changes will pass prior to committing with:

`pre-commit run --all-files`
