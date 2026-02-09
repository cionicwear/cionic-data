# GitHub Copilot Instructions for CIONIC Data Tools

Python 3.9+ biomechanical data analysis (NumPy, Pandas, SciPy, Matplotlib, pytest).

## Linting & Formatting

**ALWAYS** run `pre-commit run --all-files` before committing.
- **Black**: 88-char, single quotes (`pyproject.toml`)
- **isort**: stdlib → third-party → first-party (`pyproject.toml`)
- **Flake8**: 88-char, ignores E203/W503 (`.flake8`)
- **nbQA**: Same for notebooks

## applyTo: "cionic/*.py"
When working within the src/ directory:

TDD Requirement: Before writing any implementation code, generate test cases in tests/test_<module>.py.
Structure: Use a class-based structure for these tests.
Verification: Ensure the test covers edge cases (null values, empty strings) before providing the final function logic.

## applyTo: ["scripts/.py", "jupyter/**/.ipynb"]
### Utility Rules:
When working on scripts, helper functions, notebooks:

Conciseness: Prioritize speed and direct implementation.
Inline Comments: Use brief inline comments for complex logic instead of external documentation.

## DRY Principle

**Immediately abstract repeated logic.** Utility modules: `npz_utils`, `pod_utils`, `tools`, `dsp`, `stats`

```python
# Bad: name = suffix if prefix is None else f'{prefix}_{suffix}'
# Good: name = flat_name(prefix, suffix)
```

## Code Standards

**Naming**: `snake_case` (vars/funcs), `PascalCase` (classes), `UPPER_SNAKE_CASE` (constants)

**Type Hints**: Required for function signatures
```python
def process(data: np.ndarray, threshold: Optional[float] = None) -> List[float]:
```

**Docstrings**: Required for modules/functions with Args/Returns sections, use Google style
Example:
```python
def calculate_metrics(
    self,
    metrics: Optional[list[Metric]] = None,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Calculate selected gait metrics for all strides and save to CSV if needed.

    Args:
        metrics (list[Metric], optional): Metrics to compute.
        output_path (str, optional): Output directory path.

    Returns:
        pd.DataFrame: DataFrame of computed metrics for all strides.
    """
```

## Data Patterns

**NumPy Structured Arrays**: For biomechanical data with explicit dtypes
```python
dtype = np.dtype([('elapsed_s', 'f8'), ('x', 'f8')])
```

**NPZ Files**: Store multiple arrays with 'segments' metadata via `npz_utils`

**Processing**: Vectorized ops, SciPy for signals, chunk large datasets, use `pathlib.Path`

## Domain Specifics

- **API**: Use `get_cionic()` for requests with caching
- **Gait**: Euler angles for joints, stride-based segments, include `elapsed_s` in time-series
- **Segmentation**: Use `segmenter`/`foot_segmenter`, return arrays with `start_s`/`stop_s`

## Pre-Submission Checklist

- [ ] Tests written BEFORE implementation (TDD)
- [ ] All tests pass: `pytest -v`
- [ ] Pre-commit passes: `pre-commit run --all-files`
- [ ] No repeated logic (DRY)
- [ ] Imports sorted (isort)
- [ ] Black formatted (88-char)
- [ ] Type hints present
- [ ] Docstrings complete
