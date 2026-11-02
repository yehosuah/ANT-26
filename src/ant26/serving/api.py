"""Serve immutable benchmark outputs without fitting models in the API."""

import json
import os
import re
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query

from ant26 import __version__


def create_app(artifact_root: str | Path | None = None) -> FastAPI:
    root = Path(artifact_root or os.environ.get("ANT26_ARTIFACT_ROOT", "artifacts")).resolve()
    app = FastAPI(title="ANT-26", version=__version__)

    def invalidated(path: Path) -> bool:
        return any(
            (path / filename).exists()
            for filename in ("invalidated.json", "normalization-defect.json", "superseded.json")
        )

    def run_path(run_id: str) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,80}", run_id):
            raise HTTPException(400, "Invalid run identifier")
        path = (root / run_id).resolve()
        if path.parent != root or not path.is_dir():
            raise HTTPException(404, "Run not found")
        if invalidated(path):
            raise HTTPException(410, "Run was invalidated; use a corrected version")
        return path

    def read_json(path: Path):
        if not path.resolve().is_relative_to(root) or not path.is_file():
            raise HTTPException(404, "Artifact not available")
        try:
            return json.loads(path.read_text())
        except (ValueError, OSError) as exc:
            raise HTTPException(503, "Artifact cannot be read") from exc

    @app.get("/health")
    def health():
        return {"status": "ok", "version": __version__, "mode": "offline_artifact_replay"}

    @app.get("/runs")
    def runs():
        return {
            "runs": sorted(
                p.name
                for p in root.iterdir()
                if p.is_dir() and (p / "report.json").is_file() and not invalidated(p)
            )
            if root.exists()
            else []
        }

    @app.get("/runs/{run_id}/report")
    def report(run_id: str):
        return read_json(run_path(run_id) / "report.json")

    @app.get("/runs/{run_id}/predictions")
    def predictions(
        run_id: str,
        driver_number: int = 12,
        session_key: int | None = None,
        model: str | None = None,
        offset: int = Query(0, ge=0),
        limit: int = Query(500, ge=1, le=5000),
    ):
        rows = read_json(run_path(run_id) / "predictions.json")
        selected = [
            row
            for row in rows
            if row.get("driver_number") == driver_number
            and (session_key is None or row.get("session_key") == session_key)
            and (model is None or row.get("model") == model)
        ]
        return {
            "count": len(selected),
            "offset": offset,
            "predictions": selected[offset : offset + limit],
        }

    @app.get("/runs/{run_id}/prediction-index")
    def prediction_index(run_id: str, driver_number: int = 12):
        rows = read_json(run_path(run_id) / "predictions.json")
        entries = {
            (row["session_key"], row.get("model", "unknown"))
            for row in rows
            if row.get("driver_number") == driver_number
        }
        return {"entries": [{"session_key": key, "model": model} for key, model in sorted(entries)]}

    @app.get("/runs/{run_id}/atlas")
    def atlas(run_id: str):
        return read_json(run_path(run_id) / "atlas.json")

    @app.get("/runs/{run_id}/diagnostics")
    def diagnostics(run_id: str):
        return read_json(run_path(run_id) / "diagnostics.json")

    return app


app = create_app()
