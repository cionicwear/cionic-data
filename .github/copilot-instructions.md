# GitHub Copilot Instructions for CIONIC Data Tools

Python 3.9+ biomechanical data analysis (NumPy, Pandas, SciPy, Matplotlib, pytest).

## Linting & Formatting

**ALWAYS** run `pre-commit run --all-files` before committing.
- **Black**: 88-char, single quotes (`pyproject.toml`)
- **isort**: stdlib → third-party → first-party (`pyproject.toml`)
- **Flake8**: 88-char, ignores E203/W503 (`.flake8`)
- **nbQA**: Same for notebooks

## Test-Driven Development (TDD)

**Write tests BEFORE implementation.** Tests in `tests/test_<module>.py` using class-based structure.

```python
class TestFeature:
    def test_basic(self, mock_data):
        assert function(mock_data) == expected
```

Run: `pytest tests/test_api.py -v` or `pytest --cov=cionic`

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

**Docstrings**: Required for modules/functions with Args/Returns sections

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
