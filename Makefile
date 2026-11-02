.PHONY: install check demo smoke

install:
	uv sync --locked --group dev --extra app --no-editable

check:
	uv run --no-sync ruff check .
	uv run --no-sync ruff format --check .
	uv run --no-sync mypy
	uv run --no-sync pytest
	uv run --no-sync ant26 protocol-check

demo:
	uv run --no-sync streamlit run app/portfolio.py --server.address 127.0.0.1 --server.port 8501 --server.headless true

smoke:
	uv run --no-sync ant26 smoke --output artifacts/smoke
