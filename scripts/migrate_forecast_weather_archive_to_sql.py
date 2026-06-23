from __future__ import annotations

import platform
from pathlib import Path
import sys

platform.machine = lambda: "AMD64"

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from forecasting.data.output_sql_store import archive_forecast_weather_snapshot


def _project_root() -> Path:
    return ROOT


def _load_config(root: Path) -> dict:
    return yaml.safe_load((root / "forecasting" / "config.yaml").read_text(encoding="utf-8"))


def _weather_archive_dir(root: Path, config: dict) -> Path:
    cache_dir = Path(str((config.get("openmeteo", {}) or {}).get("cache_dir") or "weather_cache"))
    if not cache_dir.is_absolute():
        cache_dir = root / cache_dir
    return cache_dir / "forecast_weather_runs"


def _archive_entries(archive_dir: Path) -> list[dict]:
    manifest_path = archive_dir / "manifest.csv"
    if manifest_path.exists():
        manifest = pd.read_csv(manifest_path)
        entries = []
        for _, row in manifest.iterrows():
            path = Path(str(row.get("SnapshotPath") or ""))
            if not path.is_absolute():
                path = archive_dir / path
            entries.append(
                {
                    "path": path,
                    "created_at_utc": row.get("CreatedAtUTC"),
                    "source": row.get("Source", "open_meteo_forecast"),
                }
            )
        return entries

    return [
        {"path": path, "created_at_utc": None, "source": "open_meteo_forecast"}
        for path in sorted(archive_dir.glob("forecast_weather_*.csv"))
    ]


def main() -> None:
    root = _project_root()
    config = _load_config(root)
    archive_dir = _weather_archive_dir(root, config)
    if not archive_dir.exists():
        raise SystemExit(f"Forecast weather archive directory not found: {archive_dir}")

    migrated = 0
    skipped = 0
    for entry in _archive_entries(archive_dir):
        path = Path(entry["path"])
        if not path.exists():
            skipped += 1
            continue
        try:
            df = pd.read_csv(path, low_memory=False)
            snapshot_id = archive_forecast_weather_snapshot(
                config,
                df,
                source=str(entry.get("source") or "open_meteo_forecast"),
                archived_at_utc=entry.get("created_at_utc"),
            )
        except Exception as exc:
            skipped += 1
            print(f"Skipped {path}: {exc}", flush=True)
            continue
        if snapshot_id:
            migrated += 1
            print(f"Migrated {path.name} -> {snapshot_id}", flush=True)
        else:
            skipped += 1

    print(f"Forecast weather archive migration complete. Migrated={migrated}, skipped={skipped}", flush=True)


if __name__ == "__main__":
    main()
