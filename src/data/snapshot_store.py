"""
SnapshotStore
=============

Persists timestamped project risk snapshots to
``data/processed/risk_snapshots.csv`` for trend analysis.

Each call to :meth:`SnapshotStore.save_snapshot` appends one row per
project.  Missing columns (e.g. ``mcda_score`` on an ML-only save) are
stored as NaN so that downstream readers can handle mixed sources safely.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

_SNAPSHOT_COLUMNS = [
    "snapshot_id",
    "timestamp",
    "source",
    "project_id",
    "project_name",
    "risk_score",
    "risk_level",
    "mcda_score",
    "mcda_risk_level",
]


class SnapshotStore:
    """Append-only store for project risk snapshots."""

    def __init__(self, path: Optional[Path] = None) -> None:
        if path is None:
            root = Path(__file__).resolve().parents[2]
            path = root / "data" / "processed" / "risk_snapshots.csv"
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def save_snapshot(self, df: pd.DataFrame, source: str = "ml") -> None:
        """Append a timestamped snapshot of *df* to the CSV store.

        Parameters
        ----------
        df:
            DataFrame containing at least one of the recognised risk
            columns.  Extra columns are ignored; missing columns are
            filled with NaN.
        source:
            Logical source label (``"ml"`` or ``"mcda"``).
        """
        if df is None or df.empty:
            return

        now = datetime.now(timezone.utc).isoformat()
        snap_id = str(uuid.uuid4())

        rows: list[dict] = []
        for _, row in df.iterrows():
            record: dict = {
                "snapshot_id": snap_id,
                "timestamp": now,
                "source": source,
                "project_id": row.get("project_id", row.get("project_name", "")),
                "project_name": row.get("project_name", ""),
                "risk_score": row.get("risk_score", float("nan")),
                "risk_level": row.get("risk_level", ""),
                "mcda_score": row.get("mcda_score", float("nan")),
                "mcda_risk_level": row.get("mcda_risk_level", ""),
            }
            rows.append(record)

        new_df = pd.DataFrame(rows, columns=_SNAPSHOT_COLUMNS)

        if self._path.exists():
            existing = pd.read_csv(self._path, dtype=str)
            combined = pd.concat([existing, new_df], ignore_index=True)
        else:
            combined = new_df

        combined.to_csv(self._path, index=False)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def load_snapshots(self) -> pd.DataFrame:
        """Return all snapshots as a DataFrame.

        Returns an empty DataFrame with the expected columns when no
        snapshots have been saved yet.
        """
        if not self._path.exists():
            return pd.DataFrame(columns=_SNAPSHOT_COLUMNS)

        df = pd.read_csv(self._path)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        for col in ("risk_score", "mcda_score"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    def get_project_history(self, project_id: str) -> pd.DataFrame:
        """Return all snapshots for a single project, sorted by time."""
        df = self.load_snapshots()
        if df.empty:
            return df
        mask = (df["project_id"] == project_id) | (df["project_name"] == project_id)
        return df[mask].sort_values("timestamp").reset_index(drop=True)
