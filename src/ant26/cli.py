"""Explicit, reproducible stages for development data and offline replay."""

import argparse
import hashlib
import json
from pathlib import Path

from ant26.domain.protocol import load_protocol, protocol_hash


def _save(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if path.exists():
        if path.read_text() != text:
            raise ValueError(f"Refusing to overwrite an existing artifact: {path}")
        return
    with path.open("x") as output:
        output.write(text)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ant26")
    parser.add_argument("--protocol", default="configs/evaluation-protocol.json")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("protocol-check")
    audit = commands.add_parser("audit")
    audit.add_argument("--offline", action="store_true")
    audit.add_argument("--races", type=int, default=3)
    audit.add_argument("--cache-root", default="data/raw")
    audit.add_argument("--output", default="data/audit")
    audit.add_argument("--telemetry-window-seconds", type=int, default=60)
    ingest = commands.add_parser("ingest")
    ingest.add_argument("--sessions", nargs="+", type=int, default=[11396, 11388, 11731])
    ingest.add_argument("--drivers", nargs="+", type=int, default=[12, 63])
    ingest.add_argument("--cache-root", default="data/raw")
    ingest.add_argument("--progress", default="data/audit/full-trace-progress.json")
    ingest.add_argument("--chunk-seconds", type=int, default=60)
    build = commands.add_parser("build")
    build.add_argument("--cache-root", default="data/raw")
    build.add_argument("--catalog", default="data/audit/catalog.json")
    build.add_argument("--sessions", nargs="*", type=int)
    build.add_argument("--drivers", nargs="+", type=int, default=[12, 63])
    build.add_argument("--output", default="data/processed")
    build.add_argument(
        "--full-traces",
        default=None,
        help="Read complete bounded telemetry manifests from an expansion progress file",
    )
    build.add_argument(
        "--audit-samples",
        default=None,
        help="Use exact sampled telemetry queries from an audit JSON; never imply full-race coverage",
    )
    benchmark = commands.add_parser("benchmark")
    benchmark.add_argument("--examples", required=True)
    benchmark.add_argument("--catalog", default="data/audit/catalog.json")
    benchmark.add_argument("--output", default="artifacts/development")
    benchmark.add_argument("--catboost", action="store_true")
    smoke = commands.add_parser("smoke")
    smoke.add_argument("--output", default="artifacts/smoke")
    train = commands.add_parser("train")
    train.add_argument("--examples", required=True)
    train.add_argument("--fit-sessions", nargs="+", type=int, required=True)
    train.add_argument("--calibration-sessions", nargs="*", type=int, default=[])
    train.add_argument("--mode", choices=["scratch", "frozen", "finetune"], default="finetune")
    train.add_argument("--pretrain-epochs", type=int, default=3)
    train.add_argument("--forecast-epochs", type=int, default=5)
    train.add_argument("--output", required=True)
    atlas = commands.add_parser("atlas")
    atlas.add_argument("--examples", required=True)
    atlas.add_argument("--bundle", required=True)
    atlas.add_argument("--query-example", required=True)
    atlas.add_argument("--output", required=True)
    serve = commands.add_parser("serve")
    serve.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    protocol = load_protocol(args.protocol)
    if args.command == "protocol-check":
        print(
            json.dumps(
                {
                    "valid": True,
                    "protocol_sha256": protocol_hash(protocol),
                    "version": protocol["protocol_version"],
                }
            )
        )
    elif args.command == "audit":
        from ant26.ingestion import run_feasibility_audit

        result = run_feasibility_audit(
            args.cache_root,
            args.output,
            sample_races=args.races,
            offline=args.offline,
            telemetry_window_seconds=args.telemetry_window_seconds,
        )
    elif args.command == "ingest":
        from ant26.ingestion import OpenF1Client, expand_full_traces

        result = expand_full_traces(
            OpenF1Client(args.cache_root),
            session_keys=tuple(args.sessions),
            driver_numbers=tuple(args.drivers),
            progress_path=args.progress,
            chunk_seconds=args.chunk_seconds,
        )
        print(
            json.dumps(
                {
                    "sessions": {
                        key: {"status": value["status"], "rows": value["rows"]}
                        for key, value in result["sessions"].items()
                    },
                    "mexico_ingested": result["mexico_ingested"],
                }
            )
        )
        print(
            json.dumps(
                {
                    "audited_races": len(result["sample_races"]),
                    "errors": sum(len(r["errors"]) for r in result["sample_races"]),
                    "mexico_ingested": result["mexico_ingested"],
                }
            )
        )
    elif args.command == "build":
        from ant26.datasets import build_examples, quality_report
        from ant26.ingestion import OpenF1Client, ProviderError

        client = OpenF1Client(Path(args.cache_root), offline=True)
        catalog = json.loads(Path(args.catalog).read_text())
        if args.audit_samples and args.full_traces:
            raise ValueError("Choose sampled or full telemetry, not both")
        full_traces = (
            json.loads(Path(args.full_traces).read_text())["sessions"] if args.full_traces else {}
        )
        sample_queries = {}
        if args.audit_samples:
            audit_payload = json.loads(Path(args.audit_samples).read_text())
            sample_queries = {
                (r["session"]["session_key"], s["query"]["driver_number"]): s["query"]
                for r in audit_payload["sample_races"]
                for s in r["sources"]
                if s["quality"]["endpoint"] == "car_data"
            }
        examples, skipped, manifests = [], [], []
        for race in catalog["development"]:
            key = race["session_key"]
            if args.sessions and key not in args.sessions:
                continue
            for driver in args.drivers:
                try:
                    sources = {}
                    for endpoint in (
                        "laps",
                        "car_data",
                        "stints",
                        "intervals",
                        "pit",
                        "race_control",
                        "weather",
                    ):
                        params = {"session_key": key}
                        if endpoint not in ("race_control", "weather"):
                            params["driver_number"] = driver
                        if endpoint == "car_data" and args.full_traces:
                            entry = full_traces.get(f"{key}:{driver}", {})
                            if entry.get("status") != "complete":
                                raise ProviderError(
                                    "Full telemetry is not complete for this driver/session"
                                )
                            sources[endpoint] = []
                            for chunk in entry["chunks"].values():
                                sources[endpoint].extend(
                                    client.fetch(endpoint, chunk["manifest"]["params"])
                                )
                                manifests.append(dict(client.last_manifest or {}))
                            continue
                        if endpoint == "car_data" and args.audit_samples:
                            if (key, driver) not in sample_queries:
                                raise ProviderError(
                                    "No audited telemetry window for this driver/session"
                                )
                            params = sample_queries[(key, driver)]
                        sources[endpoint] = client.fetch(endpoint, params)
                        manifests.append(dict(client.last_manifest or {}))
                    examples.extend(
                        build_examples(
                            sources.pop("laps"),
                            sources.pop("car_data"),
                            protocol=protocol,
                            **sources,
                        )
                    )
                except ProviderError as exc:
                    skipped.append(
                        {"session_key": key, "driver_number": driver, "reason": str(exc)}
                    )
        if not examples:
            raise ValueError("No complete cached sessions available; run audit first")
        payload = {
            "data_kind": "real_openf1_sampled" if args.audit_samples else "real_openf1",
            "protocol_sha256": protocol_hash(protocol),
            "examples": examples,
            "quality": quality_report(examples),
            "skipped": skipped,
            "source_manifests": manifests,
        }
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        path = Path(args.output) / f"examples-{digest[:16]}.json"
        _save(path, payload)
        print(
            json.dumps(
                {
                    "path": str(path),
                    "quality": payload["quality"],
                    "skipped_driver_sessions": len(skipped),
                }
            )
        )
    elif args.command in ("benchmark", "smoke"):
        from ant26.evaluation import run_benchmark

        if args.command == "smoke":
            from ant26.demo import synthetic_examples

            examples, catalog = synthetic_examples(protocol)
            _save(
                Path(args.output) / "examples.json",
                {
                    "data_kind": catalog["data_kind"],
                    "protocol_sha256": protocol_hash(protocol),
                    "examples": examples,
                },
            )
            _save(Path(args.output) / "catalog.json", catalog)
        else:
            payload = json.loads(Path(args.examples).read_text())
            if not isinstance(payload, dict) or payload.get("protocol_sha256") != protocol_hash(
                protocol
            ):
                raise ValueError(
                    "Dataset protocol hash does not match the active protocol; rebuild inputs"
                )
            examples = payload["examples"]
            catalog = json.loads(Path(args.catalog).read_text())
            catalog["data_kind"] = (
                payload.get("data_kind", "unspecified")
                if isinstance(payload, dict)
                else "unspecified"
            )
        report = run_benchmark(
            examples,
            catalog,
            protocol=protocol,
            output_dir=args.output,
            include_catboost=getattr(args, "catboost", False),
        )
        _save(Path(args.output) / "report.json", report)
        _save(Path(args.output) / "predictions.json", report["predictions"])
        print(
            json.dumps(
                {
                    key: report[key]
                    for key in (
                        "status",
                        "synthetic",
                        "evaluated_fold_count",
                        "antonelli_test_race_count",
                        "input_example_count",
                    )
                }
            )
        )
    elif args.command in ("train", "atlas"):
        from ant26.models.baselines import valid_examples
        from ant26.models.representation import RepresentationBundle, train_representation

        payload = json.loads(Path(args.examples).read_text())
        if payload.get("protocol_sha256") != protocol_hash(protocol):
            raise ValueError("Dataset protocol hash does not match the active protocol")
        examples = payload["examples"]
        if args.command == "train":
            from ant26.evaluation.splits import validate_scope

            fitting = [r for r in examples if r["session_key"] in args.fit_sessions]
            calibration = [r for r in examples if r["session_key"] in args.calibration_sessions]
            fit_scope, calibration_scope = set(args.fit_sessions), set(args.calibration_sessions)
            if fit_scope.intersection(calibration_scope):
                raise ValueError("Fitting and calibration sessions must be disjoint")
            validate_scope(fitting, fit_scope, role="CLI fitting preflight")
            if {r["session_key"] for r in fitting} != fit_scope:
                raise ValueError("Dataset does not account for every requested fitting session")
            if calibration_scope:
                validate_scope(calibration, calibration_scope, role="CLI calibration preflight")
                if {r["session_key"] for r in calibration} != calibration_scope or len(
                    valid_examples(calibration)
                ) < 10:
                    raise ValueError(
                        "Calibration requires all declared sessions and ten valid examples"
                    )
            bundle = train_representation(
                fitting,
                set(args.fit_sessions),
                protocol=protocol,
                mode=args.mode,
                pretrain_epochs=args.pretrain_epochs,
                forecast_epochs=args.forecast_epochs,
                data_kind=payload["data_kind"],
            )
            if args.calibration_sessions:
                bundle.calibrate(calibration, calibration_sessions=calibration_scope)
            bundle.save(args.output)
            print(
                json.dumps(
                    {
                        "output": args.output,
                        "encoder_parameters": bundle.metadata["encoder_parameters"],
                        "synthetic": bundle.metadata["synthetic"],
                        "calibrated": bundle.metadata["calibrated"],
                    }
                )
            )
        else:
            from ant26.atlas import AtlasIndex

            bundle = RepresentationBundle.load(args.bundle)
            expected_bundle_hash = protocol_hash(protocol)
            if bundle.metadata["protocol_sha256"] != expected_bundle_hash:
                raise ValueError("Encoder protocol does not match the active protocol")
            fitting_sessions = set(bundle.metadata["fit_sessions"])
            fitting = valid_examples([r for r in examples if r["session_key"] in fitting_sessions])
            if {r["session_key"] for r in fitting} != fitting_sessions:
                raise ValueError("Atlas input must account for every encoder fitting session")
            fitting_hash = hashlib.sha256(
                json.dumps(fitting, sort_keys=True, allow_nan=False).encode()
            ).hexdigest()
            if fitting_hash != bundle.metadata["fitting_examples_sha256"]:
                raise ValueError("Atlas input differs from the encoder's exact fitting snapshot")
            query = next(r for r in examples if r["example_id"] == args.query_example)
            if query["session_key"] in fitting_sessions or any(
                r["cutoff"] >= query["cutoff"] for r in fitting
            ):
                raise ValueError("Atlas evaluation query must follow all encoder fitting examples")
            index = AtlasIndex().fit(
                fitting, bundle.encode(fitting), allowed_sessions=fitting_sessions
            )
            index.save(Path(args.output) / "index.json")
            result = index.query(query, bundle.encode([query])[0])
            result["encoder_fit_sessions"] = sorted(fitting_sessions)
            result["synthetic"] = bundle.metadata["synthetic"]
            _save(Path(args.output) / "atlas.json", result)
            print(
                json.dumps(
                    {
                        "output": args.output,
                        "neighbor_count": len(result["neighbors"]),
                        "synthetic": result["synthetic"],
                    }
                )
            )
    elif args.command == "serve":
        import uvicorn

        uvicorn.run("ant26.serving.api:app", host="127.0.0.1", port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
