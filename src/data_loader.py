from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class LinkedData:
    """Container for datasets shared across visualization components."""

    orbit: pd.DataFrame
    wafer: pd.DataFrame
    waveform: pd.DataFrame

    def default_selection(self) -> int:
        """Return a safe default trajectory identifier."""
        if "trajectory_id" in self.waveform.columns and not self.waveform.empty:
            return int(self.waveform["trajectory_id"].iloc[0])
        if "trajectory_id" in self.orbit.columns and not self.orbit.empty:
            return int(self.orbit["trajectory_id"].iloc[0])
        raise DataValidationError("No trajectory data available for default selection.")


class DataValidationError(RuntimeError):
    """Raised when required columns are missing from input data."""


def _resolve_path(path: Optional[str | Path]) -> Optional[Path]:
    if path is None:
        return None
    candidate = Path(path).expanduser()
    return candidate if candidate.exists() else None


def load_orbit_data(path: Optional[str | Path] = None) -> pd.DataFrame:
    """Load electron orbit data and guarantee required columns."""
    resolved = _resolve_path(path)
    if resolved:
        df = pd.read_csv(resolved)
    else:
        df = _generate_sample_orbit()

    required = {"trajectory_id", "x", "y", "z"}
    missing = required.difference(df.columns)
    if missing:
        raise DataValidationError(f"Orbit data missing columns: {sorted(missing)}")

    df = df.copy()
    if "point_id" not in df.columns:
        df["point_id"] = np.arange(len(df))

    if "segment_index" not in df.columns:
        df["segment_index"] = df.groupby("trajectory_id").cumcount()

    if "potential" not in df.columns:
        df["potential"] = np.linalg.norm(df[["x", "y", "z"]], axis=1)

    if "die_id" not in df.columns:
        df["die_id"] = df["trajectory_id"].astype(int)

    return df[
        [
            "point_id",
            "trajectory_id",
            "segment_index",
            "die_id",
            "x",
            "y",
            "z",
            "potential",
        ]
    ].sort_values(["trajectory_id", "segment_index"])


def load_wafer_data(path: Optional[str | Path] = None) -> pd.DataFrame:
    """Load wafer die metadata; create default 20x20 grid if missing."""
    resolved = _resolve_path(path)
    if resolved:
        df = pd.read_csv(resolved)
    else:
        df = _generate_sample_wafer()

    required = {"die_id", "row", "col"}
    missing = required.difference(df.columns)
    if missing:
        raise DataValidationError(f"Wafer data missing columns: {sorted(missing)}")

    df = df.copy()
    if "status" not in df.columns:
        df["status"] = "nominal"

    if "metric" not in df.columns:
        df["metric"] = np.random.normal(loc=0.0, scale=1.0, size=len(df))

    return df[["die_id", "row", "col", "status", "metric"]]


def load_waveform_data(
    path: Optional[str | Path] = None, *, orbit_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """Load trajectory-level metric data; fall back to synthetic generation."""
    resolved = _resolve_path(path)
    if resolved:
        df = pd.read_csv(resolved)
    else:
        df = _generate_sample_waveform(orbit_df)

    required = {"trajectory_id", "amplitude", "metric_b"}
    missing = required.difference(df.columns)
    if missing:
        raise DataValidationError(f"Waveform data missing columns: {sorted(missing)}")

    df = df.copy()
    if "die_id" not in df.columns and orbit_df is not None and "die_id" in orbit_df.columns:
        lookup = orbit_df.drop_duplicates("trajectory_id")[["trajectory_id", "die_id"]]
        df = df.merge(lookup, on="trajectory_id", how="left")

    if "die_id" not in df.columns:
        df["die_id"] = df["trajectory_id"].astype(int)

    if "anomaly_score" not in df.columns:
        df["anomaly_score"] = np.random.rand(len(df))

    return df[["trajectory_id", "die_id", "amplitude", "metric_b", "anomaly_score"]]


def build_linked_data(
    orbit_path: Optional[str | Path] = None,
    wafer_path: Optional[str | Path] = None,
    waveform_path: Optional[str | Path] = None,
) -> LinkedData:
    """Load and harmonise datasets for the interactive dashboards."""
    orbit_df = load_orbit_data(orbit_path)
    wafer_df = load_wafer_data(wafer_path)
    waveform_df = load_waveform_data(waveform_path, orbit_df=orbit_df)

    orbit_df = orbit_df.merge(
        wafer_df[["die_id", "row", "col", "status", "metric"]],
        on="die_id",
        how="left",
    )

    if orbit_df[["row", "col"]].isna().any().any():
        missing_die_ids = orbit_df.loc[orbit_df["row"].isna(), "die_id"].unique()
        raise DataValidationError(
            f"Missing wafer metadata for die_id(s): {','.join(map(str, missing_die_ids))}"
        )

    waveform_df = waveform_df.merge(
        wafer_df[["die_id", "row", "col"]],
        on="die_id",
        how="left",
    )

    return LinkedData(orbit=orbit_df, wafer=wafer_df, waveform=waveform_df)


def export_sample_payload(path: str | Path) -> None:
    """Dump the synthetic dataset payload as JSON for inspection."""
    bundle = build_linked_data()
    payload = {
        "orbit": bundle.orbit.to_dict(orient="records"),
        "wafer": bundle.wafer.to_dict(orient="records"),
        "waveform": bundle.waveform.to_dict(orient="records"),
    }
    Path(path).write_text(json.dumps(payload, indent=2))


def _generate_sample_orbit(
    trajectories: int = 400, points_per_traj: int = 40
) -> pd.DataFrame:
    """Create synthetic trajectories grouped by trajectory_id for development."""
    trajectories = max(1, trajectories)
    points_per_traj = max(10, points_per_traj)
    records = []
    for traj in range(trajectories):
        t = np.linspace(0, 1, points_per_traj)
        base_angle = np.random.uniform(0, 2 * np.pi)
        radius = np.random.uniform(0.3, 1.0)
        twist = np.random.uniform(1.0, 3.5)
        elevation = np.random.uniform(3.0, 8.0)
        x = np.cos(base_angle + twist * t) * radius
        y = np.sin(base_angle + twist * t) * radius
        z = elevation * t + np.sin(t * 4 * np.pi) * 0.3
        drift = np.random.uniform(-0.5, 0.5, size=3)
        x = x + drift[0]
        y = y + drift[1]
        z = z + drift[2]
        potential = np.linspace(
            np.random.uniform(50, 120), np.random.uniform(80, 150), points_per_traj
        )
        die_id = traj
        for idx, (xx, yy, zz, pot) in enumerate(zip(x, y, z, potential)):
            point_id = traj * points_per_traj + idx
            records.append(
                {
                    "point_id": point_id,
                    "trajectory_id": traj,
                    "die_id": die_id,
                    "x": xx,
                    "y": yy,
                    "z": zz,
                    "potential": pot,
                }
            )
    return pd.DataFrame.from_records(records)


def _generate_sample_wafer(size: int = 20) -> pd.DataFrame:
    """Generate a 20x20 wafer map with baseline metrics."""
    rows = []
    for row in range(size):
        for col in range(size):
            die_id = row * size + col
            base = np.sin(row / size * np.pi) + np.cos(col / size * np.pi)
            rows.append(
                {
                    "die_id": die_id,
                    "row": row,
                    "col": col,
                    "status": "nominal",
                    "metric": base + np.random.normal(0, 0.1),
                }
            )
    return pd.DataFrame(rows)


def _generate_sample_waveform(
    orbit_df: Optional[pd.DataFrame], trajectories: int = 400
) -> pd.DataFrame:
    """Generate trajectory-level scatter metrics for initial development."""
    if orbit_df is None or orbit_df.empty:
        orbit_df = _generate_sample_orbit(trajectories=trajectories)

    representatives = (
        orbit_df[["trajectory_id", "die_id"]]
        .drop_duplicates("trajectory_id")
        .sort_values("trajectory_id")
    )

    rng = np.random.default_rng(seed=42)
    data = []
    for _, row in representatives.iterrows():
        traj_id = int(row["trajectory_id"])
        die_id = int(row["die_id"])
        amplitude = rng.normal(loc=0.0, scale=1.0)
        metric_b = rng.normal(loc=0.0, scale=1.0)
        anomaly_score = abs(amplitude) * 0.3 + abs(metric_b) * 0.2 + rng.random() * 0.2
        data.append(
            {
                "trajectory_id": traj_id,
                "die_id": die_id,
                "amplitude": amplitude,
                "metric_b": metric_b,
                "anomaly_score": anomaly_score,
            }
        )
    return pd.DataFrame(data)
