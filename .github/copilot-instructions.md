# GitHub Copilot Instructions for CIONIC Data Tools

This document provides coding guidelines and best practices for developing in the CIONIC Data Tools repository. Follow these instructions to maintain consistency with the existing codebase.

## Technology Stack

- **Language**: Python 3.9+
- **Core Libraries**: NumPy, Pandas, SciPy, Matplotlib, Requests
- **Testing Framework**: pytest
- **Primary Domain**: Biomechanical data analysis, IMU data processing, gait analysis

## Code Style and Formatting

### Linting and Formatting Tools

All code must pass pre-commit hooks before commits are accepted. The following tools are configured and enforced:

1. **Black** (Code Formatter)
   - Line length: 88 characters
   - Target Python version: 3.9
   - Skip string normalization (use single quotes for strings)
   - Configuration: `pyproject.toml`

2. **isort** (Import Sorting)
   - Profile: black-compatible
   - Line length: 88 characters
   - Known first-party modules: `cionic`, `kinematics`
   - Sort imports in the following order: standard library, third-party, first-party
   - Configuration: `pyproject.toml`

3. **Flake8** (Linter)
   - Max line length: 88 characters
   - Extended ignores: E203 (whitespace before ':'), W503 (line break before binary operator)
   - Excludes: `.git`, `__pycache__`, `venv`, `.ipynb_checkpoints`, `__init__.py`
   - Configuration: `.flake8`

4. **nbQA** (Notebook Quality Assurance)
   - Applies Black, isort, and Flake8 to Jupyter notebooks
   - Same standards as Python files

### Pre-commit Workflow

- **ALWAYS** run `pre-commit run --all-files` before committing to catch formatting and linting issues early
- Pre-commit hooks will automatically run on `git commit` and block commits that fail checks
- If hooks fail, fix the issues and re-stage the files before committing again

### Import Organization

Follow this import order (enforced by isort):

```python
# Standard library imports
import os
import sys
from typing import Optional

# Third-party imports
import numpy as np
import pandas as pd
import requests

# First-party imports
from cionic import api, kinematics, npz_utils
```

### String Formatting

- Use single quotes for strings: `'example'` not `"example"`
- Use f-strings for string formatting: `f'Value: {variable}'`
- Skip string normalization as per Black configuration

## Test-Driven Development (TDD)

### TDD Requirement

**ALWAYS write unit and integration tests BEFORE writing the actual function or component logic.**

1. **Start with tests**: Define expected behavior and edge cases in tests first
2. **Write minimal code**: Implement only what's needed to pass the tests
3. **Refactor**: Clean up code while ensuring tests still pass

### Test Organization

- **Test files**: Located in `tests/` directory
- **Naming convention**: `test_<module_name>.py` (e.g., `test_api.py`, `test_gait_metrics.py`)
- **Test structure**: Use class-based organization with `Test<FeatureName>` classes
- **Test methods**: Use descriptive names like `test_<feature>_<scenario>` (e.g., `test_include_eulers_basic`)

### Test Patterns

```python
"""Module docstring with pytest examples.

pytest tests/test_api.py -v
pytest tests/test_api.py::TestIncludeEulersToNpz -v  # Run specific class
pytest tests/test_api.py --cov=cionic.api --cov-report=html  # With coverage
"""

import pytest
from unittest.mock import patch, Mock

from cionic import api


@pytest.fixture
def mock_data():
    """Create mock data for testing."""
    # Setup mock data
    return mock_data


class TestFeatureName:
    """Test class for specific feature."""
    
    def test_basic_functionality(self, mock_data):
        """Test basic functionality with expected inputs."""
        # Arrange
        expected = ...
        
        # Act
        result = function(mock_data)
        
        # Assert
        assert result == expected
    
    def test_edge_case(self):
        """Test edge case behavior."""
        # Test implementation
        pass
```

### Test Fixtures

- Use pytest fixtures for reusable test data and setup
- Store complex test data in `tests/fixtures/` directory
- Create fixtures that represent realistic biomechanical data (NumPy structured arrays)
- Use `@pytest.fixture` decorator for setup code

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_api.py -v

# Run specific test class
pytest tests/test_api.py::TestClassName -v

# Run with coverage
pytest --cov=cionic --cov-report=html
```

## DRY Principle (Don't Repeat Yourself)

### Strict DRY Adherence

**Immediately abstract repeated logic into utility functions.** If you find yourself copying code, stop and refactor.

### When to Abstract

- **Repeated code blocks**: If code appears more than once, create a utility function
- **Similar logic**: If logic is similar but slightly different, parameterize the differences
- **Complex operations**: If an operation has multiple steps, encapsulate it in a function

### Utility Function Guidelines

- Create utility functions in appropriate modules:
  - `cionic/npz_utils.py`: NPZ file operations
  - `cionic/pod_utils.py`: POD (wearable) device utilities
  - `cionic/tools.py`: General-purpose utilities
  - `cionic/dsp.py`: Digital signal processing utilities
  - `cionic/stats.py`: Statistical functions
  
- Use clear, descriptive function names
- Add docstrings explaining purpose, parameters, and return values
- Keep functions focused on a single responsibility

### Example of DRY Refactoring

**Bad (Repeated Code):**
```python
# In multiple places
if prefix is None:
    name = suffix
else:
    name = f'{prefix}_{suffix}'
```

**Good (Abstracted):**
```python
def flat_name(prefix, suffix):
    """Create a flattened name from prefix and suffix."""
    if prefix is None:
        return suffix
    else:
        return f'{prefix}_{suffix}'

# In multiple places
name = flat_name(prefix, suffix)
```

## Documentation

### Module Docstrings

Every module should have a comprehensive docstring at the top:

```python
"""
Module purpose and functionality description.

Detailed explanation of what the module does, typical use cases,
and any important notes about data types or processing.

Usage Example (CLI):
    python3 -m cionic.module_name

Usage Example (Code):
    from cionic import module_name
    
    result = module_name.function(args)

Note on data types:
    Explain expected input/output data structures, especially for
    NumPy structured arrays and Pandas DataFrames.

Output:
    Description of output format and structure.
"""
```

### Function Docstrings

Use clear, concise docstrings for functions:

```python
def function_name(param1, param2):
    """Brief one-line description.
    
    More detailed explanation if needed.
    
    Args:
        param1: Description of param1
        param2: Description of param2
    
    Returns:
        Description of return value
    """
```

### Comments

- Use comments sparingly; prefer self-documenting code
- Comment complex algorithms or non-obvious logic
- Explain "why" not "what" (code should show what it does)

## NumPy and Data Processing

### Structured Arrays

- Use NumPy structured arrays (record arrays) for data with multiple fields
- Define dtypes explicitly for clarity and type safety
- Prefer structured arrays over DataFrames for memory efficiency in large datasets

```python
# Define dtype
dtype = np.dtype([
    ('elapsed_s', 'f8'),
    ('x', 'f8'),
    ('y', 'f8'),
    ('z', 'f8'),
])

# Create structured array
data = np.array([(0.0, 1.0, 2.0, 3.0)], dtype=dtype)

# Access fields
elapsed_time = data['elapsed_s']
```

### Data Processing Patterns

- Use vectorized NumPy operations instead of loops when possible
- Leverage NumPy broadcasting for element-wise operations
- Use SciPy for signal processing (filtering, interpolation, etc.)
- Process data in chunks if dealing with large datasets

## File Operations

### NPZ Files

- Use NPZ format for storing multiple NumPy arrays in a single file
- Include a 'segments' array to describe the structure of data streams
- Save metadata alongside data arrays
- Use `npz_utils` module for NPZ operations

### Path Handling

- Use `pathlib.Path` for cross-platform path operations
- Ensure parent directories exist before writing files:

```python
import pathlib

def ensure_parent(path):
    """Ensure parent directory exists for given path."""
    path = pathlib.Path(path)
    try:
        path.parent.mkdir(parents=True)
    except FileExistsError:
        pass
    
    assert path.parent.exists()
    assert path.parent.is_dir()
    
    return path
```

## Error Handling

### Assertions vs Exceptions

- Use assertions for internal invariants that should never fail
- Use exceptions for expected error conditions
- Provide informative error messages

```python
# Good: Descriptive assertion
assert stream_name in valid_streams, f"Invalid stream: {stream_name}"

# Good: Informative exception
if not file.exists():
    raise FileNotFoundError(f"NPZ file not found: {file}")
```

### HTTP Requests

- Check status codes and provide helpful error messages
- Log API requests and responses for debugging
- Use caching where appropriate to reduce API calls

## Type Hints

- Use type hints for function parameters and return values
- Import from `typing` module when needed: `Optional`, `List`, `Dict`, etc.
- Type hints improve code clarity and enable better IDE support

```python
from typing import Optional, List

def process_data(data: np.ndarray, threshold: Optional[float] = None) -> List[float]:
    """Process data with optional threshold."""
    # Implementation
    pass
```

## Naming Conventions

### Variables and Functions

- Use `snake_case` for variables and functions: `stride_time`, `calculate_metrics`
- Use descriptive names that indicate purpose: `elapsed_s` (elapsed seconds), not `e`
- Avoid single-letter variables except in mathematical contexts (e.g., `i`, `j`, `k` for loop indices)

### Classes

- Use `PascalCase` for class names: `GaitMetricsCalculator`, `FootSegmenter`
- Use descriptive names that indicate the class's purpose

### Constants

- Use `UPPER_SNAKE_CASE` for module-level constants
- Define constants at the top of the module

### Files and Modules

- Use `snake_case` for file and module names: `gait_metrics.py`, `npz_utils.py`
- Module names should be short and descriptive

## Repository-Specific Patterns

### API Module

- Use `get_cionic()` for API requests with caching support
- Store authentication token in `authtoken` variable
- Use `ensure_parent()` to create parent directories before saving files

### Kinematics and Gait Analysis

- Use Euler angles for joint rotations
- Process data in stride-based segments for gait analysis
- Include elapsed time (`elapsed_s`) in all time-series data

### Segmentation

- Use `segmenter` module for identifying strides and gait events
- Use `foot_segmenter` for foot-specific segmentation logic
- Return segment boundaries as structured arrays with `start_s` and `stop_s` fields

## Git and Version Control

### Commit Messages

- Use clear, descriptive commit messages
- Start with a verb in present tense: "Add", "Fix", "Update", "Refactor"
- Keep first line under 50 characters; add details in subsequent lines if needed

### Branches

- Create feature branches for new development
- Use descriptive branch names: `feature/gait-metrics`, `fix/segmentation-bug`

## Summary Checklist

Before submitting code, ensure:

- [ ] Tests are written BEFORE implementation (TDD)
- [ ] All tests pass: `pytest -v`
- [ ] Pre-commit hooks pass: `pre-commit run --all-files`
- [ ] Code follows DRY principle (no repeated logic)
- [ ] Imports are organized correctly (isort)
- [ ] Code is formatted with Black (88-char line length)
- [ ] Type hints are used for function signatures
- [ ] Docstrings are present for modules and functions
- [ ] NumPy structured arrays are used appropriately for data
- [ ] Error handling is appropriate and informative
- [ ] Code is consistent with existing patterns in the repository
