"""Small read-only OpenF1 client with content-addressed immutable snapshots."""

from __future__ import annotations

import csv
import fcntl
import hashlib
import io
import json
import os
import re
import tempfile
import time
from datetime import datetime, timedelta, timezone
from http.client import IncompleteRead
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


class ProviderError(RuntimeError):
    """Provider access or response schema failed."""


class CacheIntegrityError(ProviderError):
    """An existing snapshot no longer matches its checksum."""


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as temp:
        temp.write(_json_bytes(value))
        temp_path = Path(temp.name)
    os.replace(temp_path, path)


class OpenF1Client:
    """Fetch lists of source rows, preserving original response bytes and manifests.

    A mutable latest-pointer selects a content-addressed snapshot. Refreshing never
    overwrites an existing snapshot. The shared lock serializes rate-budget updates
    across clients/processes using the same cache root. An offline miss fails visibly.
    """

    def __init__(
        self,
        cache_root: Path | str = Path("data/raw"),
        *,
        base_url: str = "https://api.openf1.org/v1",
        offline: bool = False,
        rate_limit_per_minute: int = 25,
        timeout: float = 45,
        max_attempts: int = 3,
        opener: Any = None,
        sleeper: Any = time.sleep,
    ) -> None:
        if not 0 <= rate_limit_per_minute <= 25:
            raise ValueError("rate budget must be between 0 (test-only) and 25 per minute")
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        self.cache_root = Path(cache_root)
        self.base_url = base_url.rstrip("/")
        self.offline = offline
        self.rate_limit_per_minute = rate_limit_per_minute
        self.timeout = timeout
        self.max_attempts = max_attempts
        self._opener = opener or urlopen
        self._sleep = sleeper
        self.last_manifest: dict[str, Any] | None = None
        self.last_window_manifests: list[dict[str, Any]] = []

    def _paths(self, endpoint: str, params: dict[str, Any]) -> tuple[str, Path]:
        if not re.fullmatch(r"[a-z_]+", endpoint):
            raise ValueError("invalid endpoint name")
        key = hashlib.sha256(_json_bytes({"base_url": self.base_url, "params": params})).hexdigest()
        return key, self.cache_root / endpoint / key

    def _throttle(self) -> None:
        if not self.rate_limit_per_minute:
            return
        self.cache_root.mkdir(parents=True, exist_ok=True)
        with (self.cache_root / ".rate.lock").open("a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            state_path = self.cache_root / ".rate.json"
            try:
                previous = float(json.loads(state_path.read_text())["last_request"])
            except (FileNotFoundError, ValueError, KeyError):
                previous = 0.0
            delay = max(0.0, previous + 60.01 / self.rate_limit_per_minute - time.time())
            if delay:
                self._sleep(delay)
            _atomic_json(state_path, {"last_request": time.time()})

    @staticmethod
    def _validate(payload: bytes) -> list[dict[str, Any]]:
        try:
            rows = json.loads(payload)
        except (ValueError, UnicodeDecodeError) as exc:
            raise ProviderError("provider did not return valid JSON") from exc
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            raise ProviderError("expected a JSON list of objects")
        return rows

    def fetch(
        self,
        endpoint: str,
        params: dict[str, Any] | None = None,
        *,
        refresh: bool = False,
        response_format: str = "json",
    ) -> list[dict[str, Any]]:
        """Return validated rows; last_manifest identifies the exact source snapshot."""
        self.last_manifest = None
        if response_format not in {"json", "csv"}:
            raise ValueError("response_format must be json or csv")
        if response_format == "csv" and endpoint != "car_data":
            raise ValueError("CSV normalization is implemented only for car_data")
        if response_format == "csv":
            params = (params or {}) | {"csv": "true"}
        params = dict(sorted((params or {}).items()))
        query_key, query_dir = self._paths(endpoint, params)
        pointer = query_dir / "latest.json"
        if pointer.exists() and not refresh:
            digest = json.loads(pointer.read_text())["sha256"]
            if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise CacheIntegrityError("invalid snapshot pointer")
            snapshot = query_dir / digest
            payload = (snapshot / "response.json").read_bytes()
            if hashlib.sha256(payload).hexdigest() != digest:
                raise CacheIntegrityError(f"snapshot checksum mismatch for {endpoint}/{query_key}")
            manifest = json.loads((snapshot / "manifest.json").read_text())
            if manifest.get("sha256") != digest or manifest.get("params") != params:
                raise CacheIntegrityError(f"snapshot manifest mismatch for {endpoint}/{query_key}")
            if manifest.get("raw_response_filename"):
                original = (snapshot / manifest["raw_response_filename"]).read_bytes()
                if hashlib.sha256(original).hexdigest() != manifest["raw_response_sha256"]:
                    raise CacheIntegrityError("original CSV response checksum mismatch")
            self.last_manifest = manifest | {"snapshot_path": str(snapshot), "cache_hit": True}
            return self._validate(payload)
        if self.offline:
            raise ProviderError(f"offline cache miss: {endpoint} {params}")
        url = f"{self.base_url}/{endpoint}"
        if params:
            url += "?" + urlencode(params)
        request = Request(url, headers={"User-Agent": "ANT-26/0.1 research (read-only)"})
        for attempt in range(self.max_attempts):
            self._throttle()
            try:
                with self._opener(request, timeout=self.timeout) as response:
                    payload = response.read()
                break
            except HTTPError as exc:
                if (
                    exc.code not in {408, 429, 500, 502, 503, 504}
                    or attempt + 1 == self.max_attempts
                ):
                    raise ProviderError(f"OpenF1 HTTP {exc.code} for {endpoint}") from exc
                try:
                    delay = min(60.0, max(float(exc.headers.get("Retry-After", 0)), 2**attempt))
                except (ValueError, AttributeError):
                    delay = 2**attempt
                self._sleep(delay)
            except (URLError, TimeoutError, OSError, IncompleteRead) as exc:
                if attempt + 1 == self.max_attempts:
                    raise ProviderError(f"OpenF1 network failure for {endpoint}: {exc}") from exc
                self._sleep(2**attempt)
        raw_payload = payload
        if response_format == "csv":
            try:
                reader = csv.DictReader(io.StringIO(payload.decode("utf-8-sig")))
                required = {
                    "date",
                    "speed",
                    "throttle",
                    "brake",
                    "rpm",
                    "n_gear",
                    "session_key",
                    "driver_number",
                }
                if not required.issubset(reader.fieldnames or []):
                    raise ValueError("required telemetry columns missing")
                rows = []
                for raw_row in reader:
                    row = {}
                    for field, value in raw_row.items():
                        if value in {"", "null", "None"}:
                            row[field] = None
                        elif field == "date":
                            timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
                            if timestamp.tzinfo is None:
                                raise ValueError("CSV dates must be timezone-aware")
                            row[field] = timestamp.astimezone(timezone.utc).isoformat()
                        else:
                            number = float(value)
                            row[field] = int(number) if number.is_integer() else number
                    rows.append(row)
                payload = _json_bytes(rows)
            except (csv.Error, UnicodeDecodeError, ValueError, TypeError) as exc:
                raise ProviderError(f"invalid telemetry CSV: {exc}") from exc
        else:
            rows = self._validate(payload)
        digest = hashlib.sha256(payload).hexdigest()
        snapshot = query_dir / digest
        manifest = {
            "provider": "openf1",
            "endpoint": endpoint,
            "params": params,
            "url": url,
            "query_key": query_key,
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "schema_version": "openf1-raw-v1",
            "sha256": digest,
            "row_count": len(rows),
            "response_format": response_format,
        }
        if response_format == "csv":
            manifest.update(
                {
                    "raw_response_filename": "response.csv",
                    "raw_response_sha256": hashlib.sha256(raw_payload).hexdigest(),
                    "normalization_schema": "car-data-csv-numeric-utc-v1",
                }
            )
        query_dir.mkdir(parents=True, exist_ok=True)
        # Lock writers so concurrent identical fetches cannot partially replace a snapshot.
        with (query_dir / ".write.lock").open("a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            if not snapshot.exists():
                temporary = Path(tempfile.mkdtemp(dir=query_dir, prefix=".snapshot-"))
                (temporary / "response.json").write_bytes(payload)
                if response_format == "csv":
                    (temporary / "response.csv").write_bytes(raw_payload)
                (temporary / "manifest.json").write_bytes(_json_bytes(manifest))
                os.rename(temporary, snapshot)
            else:
                stored = (snapshot / "response.json").read_bytes()
                if hashlib.sha256(stored).hexdigest() != digest:
                    raise CacheIntegrityError(
                        f"existing content-addressed snapshot corrupted: {digest}"
                    )
                manifest = json.loads((snapshot / "manifest.json").read_text())
            _atomic_json(pointer, {"sha256": digest})
        self.last_manifest = manifest | {"snapshot_path": str(snapshot), "cache_hit": False}
        return rows

    def fetch_time_range(
        self,
        endpoint: str,
        params: dict[str, Any],
        start: str | datetime,
        end: str | datetime,
        *,
        chunk_seconds: int = 60,
    ) -> list[dict[str, Any]]:
        """Fetch a half-open UTC interval in bounded snapshots, preserving source rows.

        Each query includes its left boundary by subtracting one microsecond from
        the API's strict greater-than filter. Source duplicate rows are retained.
        The result covers [start,end); last_window_manifests records every chunk.
        """

        def parse(value: str | datetime) -> datetime:
            result = (
                datetime.fromisoformat(value.replace("Z", "+00:00"))
                if isinstance(value, str)
                else value
            )
            if result.tzinfo is None:
                raise ValueError("time ranges must be timezone-aware")
            return result.astimezone(timezone.utc)

        left, stop = parse(start), parse(end)
        if chunk_seconds <= 0 or left >= stop:
            raise ValueError("time range and chunk size must be positive")
        rows: list[dict[str, Any]] = []
        manifests = []
        self.last_window_manifests = manifests
        while left < stop:
            right = min(left + timedelta(seconds=chunk_seconds), stop)
            query = params | {
                "date>": (left - timedelta(microseconds=1)).replace(tzinfo=None).isoformat(),
                "date<": right.replace(tzinfo=None).isoformat(),
            }
            chunk = self.fetch(endpoint, query)
            # Reject a provider that silently ignores filters rather than duplicating full sessions.
            for row in chunk:
                if "date" not in row or not left <= parse(row["date"]) < right:
                    raise ProviderError(
                        "provider returned a row outside the requested UTC interval"
                    )
            rows.extend(chunk)
            manifests.append(dict(self.last_manifest or {}))
            left = right
        self.last_window_manifests = manifests
        return rows
