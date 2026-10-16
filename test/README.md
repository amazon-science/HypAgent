This package uses PyTest (http://pytest.org/) for its tests.

## Writing tests

Place test files in this directory, using file names that start with `test_`.

## Running tests

From the repository root:

```
$ pytest
```

`pytest` reads its configuration from `pyproject.toml` and puts `src/` on the
import path, so no install step is needed. Any pytest option can be passed
directly; for example, to run a single test or a subset of tests:

```
$ pytest -k TEST_PATTERN
```

To debug failing tests:

```
$ pytest --pdb
```

This will drop you into the Python debugger on the failed test.

### Importing tests/fixtures

The `test` module is generally not directly importable and it's generally acceptable to use relative imports inside test cases.

### Fixtures

Pytest provides `conftest.py` as a mechanism to store test fixtures.  However, there may be times when it makes sense to include a `test/fixtures` module to locate complex or large fixtures.

### Common Errors

#### ModuleNotFoundError: No module named "test.fixtures"

The `test` and sometimes `test/fixtures` modules need to be importable.  To allow these to be importable, create a `__init__.py` file in each directory.
- `test/__init__.py`
- `test/fixtures/__init__.py` (optional)
