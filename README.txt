wordrobe fixes

Apply from the repository root with:

    git apply wordrobe-fixes.patch

Validation performed in the sandbox:
- python -m compileall -q src tests
- PYTHONPATH=src pytest -q tests/component -k 'not installed_entrypoint'
  -> 42 passed, 2 deselected
- PYTHONPATH=src python -m wordrobe segment word2number
  -> word 2 number

The two deselected tests require an installed console-script entry point, which the sandbox does not have.
The repository requires Python >=3.14, while the sandbox has Python 3.13, so the uv-managed full `just check` gate could not be run offline.
