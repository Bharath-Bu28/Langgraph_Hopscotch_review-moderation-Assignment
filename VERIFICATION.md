# Verification

- Python 3.12 is the requirements-lock snapshot; the project declares Python 3.11 or newer.
- Fresh Python 3.14.8 environment: `pip install -e ".[dev,ui]"` succeeded.
- `python -m pytest -q`: 23 passed.
- `python -m ruff check src tests`: all checks passed.
- Offline baseline CLI: all 7 fixture reviews matched the required outcomes.
- Streamlit viewer started successfully; its health endpoint returned `ok`.
- Live provider calls not run; they require valid provider credentials.
