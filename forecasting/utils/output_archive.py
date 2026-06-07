from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd


MANIFEST_NAME = "manifest.csv"


def _snapshot_hash(df: pd.DataFrame, hash_columns: list[str] | None = None) -> str:
    if df is None or df.empty:
        return ""
    cols = [c for c in (hash_columns or list(df.columns)) if c in df.columns]
    if not cols:
        cols = list(df.columns)
    work = df[cols].copy()
    if "DT" in work.columns:
        work["DT"] = pd.to_datetime(work["DT"], errors="coerce", utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        work.sort_values("DT", inplace=True)
    work = work.reset_index(drop=True)
    payload = work.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _dt_bounds(df: pd.DataFrame) -> tuple[str, str]:
    if df is None or df.empty or "DT" not in df.columns:
        return "", ""
    dt = pd.to_datetime(df["DT"], errors="coerce", utc=True)
    if dt.dropna().empty:
        return "", ""
    return dt.min().isoformat(), dt.max().isoformat()


def _read_manifest(archive_dir: Path) -> pd.DataFrame:
    path = archive_dir / MANIFEST_NAME
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def save_distinct_snapshot(
    df: pd.DataFrame,
    archive_dir: Path,
    stem: str,
    hash_columns: list[str] | None = None,
    metadata: dict | None = None,
) -> Path | None:
    if df is None or df.empty:
        return None
    archive_dir.mkdir(parents=True, exist_ok=True)
    digest = _snapshot_hash(df, hash_columns=hash_columns)
    if not digest:
        return None

    manifest = _read_manifest(archive_dir)
    if not manifest.empty and "ContentHash" in manifest.columns:
        existing = manifest[manifest["ContentHash"].astype(str).eq(digest)]
        if not existing.empty and "SnapshotPath" in existing.columns:
            existing_path = Path(str(existing.iloc[-1]["SnapshotPath"]))
            if existing_path.exists():
                return existing_path

    created_at = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")
    path = archive_dir / f"{stem}_{created_at}_{digest[:10]}.csv"
    df.to_csv(path, index=False)

    first_dt, last_dt = _dt_bounds(df)
    row = {
        "CreatedAtUTC": pd.Timestamp.utcnow().isoformat(),
        "Stem": stem,
        "SnapshotPath": str(path),
        "ContentHash": digest,
        "Rows": int(len(df)),
        "FirstDT": first_dt,
        "LastDT": last_dt,
    }
    for key, value in (metadata or {}).items():
        row[str(key)] = value

    manifest = pd.concat([manifest, pd.DataFrame([row])], ignore_index=True, sort=False)
    manifest.to_csv(archive_dir / MANIFEST_NAME, index=False)
    return path


def load_latest_distinct_snapshot(
    archive_dir: Path,
    current_df: pd.DataFrame | None = None,
    hash_columns: list[str] | None = None,
) -> pd.DataFrame:
    manifest = _read_manifest(archive_dir)
    if manifest.empty or "SnapshotPath" not in manifest.columns:
        return pd.DataFrame()

    current_hash = _snapshot_hash(current_df, hash_columns=hash_columns) if current_df is not None and not current_df.empty else ""
    if "CreatedAtUTC" in manifest.columns:
        manifest["_CreatedAtUTC"] = pd.to_datetime(manifest["CreatedAtUTC"], errors="coerce", utc=True)
        manifest.sort_values("_CreatedAtUTC", ascending=False, inplace=True)
    else:
        manifest = manifest.iloc[::-1].copy()

    for _, row in manifest.iterrows():
        if current_hash and str(row.get("ContentHash", "")) == current_hash:
            continue
        path = Path(str(row.get("SnapshotPath", "")))
        if not path.exists():
            continue
        try:
            return pd.read_csv(path, low_memory=False)
        except Exception:
            continue
    return pd.DataFrame()
