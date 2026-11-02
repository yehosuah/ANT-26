import json

import ant26.ingestion
from ant26.cli import main


def test_audit_and_ingestion_have_separate_result_contracts(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(
        ant26.ingestion,
        "run_feasibility_audit",
        lambda *a, **kw: {"sample_races": [{"errors": []}], "mexico_ingested": False},
    )
    assert main(["audit", "--offline", "--output", str(tmp_path)]) == 0
    assert json.loads(capsys.readouterr().out)["audited_races"] == 1
    monkeypatch.setattr(
        ant26.ingestion,
        "expand_full_traces",
        lambda *a, **kw: {
            "sessions": {"11396:12": {"status": "complete", "rows": 100}},
            "mexico_ingested": False,
        },
    )
    assert main(["ingest", "--format", "json", "--progress", str(tmp_path / "progress.json")]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["sessions"]["11396:12"]["rows"] == 100
    assert not result["mexico_ingested"]
