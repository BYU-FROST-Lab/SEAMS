## Python

Multisession development package. Requires **Python 3.12+**.

### Setup

1. Create and activate a virtual environment (conda, mamba, venv, etc.):

```bash
mamba create -n multisession python=3.12 pip
mamba activate multisession
```

2. From the repository root, install dependencies (including the custom GTSAM wheel in `environment/`):

```bash
pip install -r requirements.txt
```

The GTSAM wheel is a pre-built, portable binary built from the Docker setup in `environment/`. It is not available on PyPI. If you rebuild the wheel, update the filename in `requirements.txt` to match the new file in `environment/`. See `environment/README.md` for how to build it.

3. Install this package in editable mode so local imports resolve:

```bash
pip install -e .
```

4. Verify GTSAM installed correctly:

```bash
python -c "import gtsam; print(gtsam.__file__)"
```

### VS Code / Cursor type hints for GTSAM

The custom GTSAM wheel does not ship type stubs. Stub files live in `environment/typings/`. The workspace `.vscode/settings.json` configures Pylance to use them:

```json
{
    "python.analysis.stubPath": "./environment/typings",
    "python.analysis.typeCheckingMode": "basic",
    "python.analysis.extraPaths": ["."]
}
```

- `stubPath` — tells Pylance where to find `.pyi` stubs for `gtsam`.
- `extraPaths` — helps resolve imports from the repo root (e.g. `multiSession`, `common`).
- `typeCheckingMode` — enables basic static checking; set to `"off"` if you only want completions without diagnostics.

Open the repo root as the workspace folder so these relative paths resolve. Reload the window after changing settings if completions do not appear immediately.

### Notebooks

To run `tests/Tests.ipynb` or `sparcification/testNFR.ipynb`, install Jupyter in your environment:

```bash
pip install jupyter ipykernel
```
