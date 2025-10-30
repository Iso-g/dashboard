from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Type
from collections import OrderedDict

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.colors import qualitative
from dash import Dash, Input, Output, State, dcc, html, ctx

from src.data_loader import build_linked_data


# ---------- Figure construction helpers -----------------------------------------------------
def make_orbit_figure(
    orbit_df: pd.DataFrame,
    selected_trajectory_ids: Optional[List[int]],
) -> go.Figure:
    """Render 3D electron orbit; emphasise trajectories tied to the current selection."""
    fig = go.Figure()

    if not orbit_df.empty:
        fig.add_trace(
            go.Scatter3d(
                x=orbit_df["x"],
                y=orbit_df["y"],
                z=orbit_df["z"],
                mode="markers",
                marker={"size": 2, "color": "#adb5bd"},
                opacity=0.2,
                hoverinfo="skip",
                name="Context",
            )
        )

    if (
        selected_trajectory_ids
        and "trajectory_id" in orbit_df.columns
    ):
        palette = qualitative.Dark24 + qualitative.Light24
        for idx, trajectory_id in enumerate(selected_trajectory_ids):
            if trajectory_id not in orbit_df["trajectory_id"].values:
                continue
            trajectory_df = orbit_df.loc[
                orbit_df["trajectory_id"] == trajectory_id
            ].sort_values("segment_index")
            color = palette[idx % len(palette)]
            fig.add_trace(
                go.Scatter3d(
                    x=trajectory_df["x"],
                    y=trajectory_df["y"],
                    z=trajectory_df["z"],
                    mode="lines+markers",
                    marker={
                        "size": 6,
                        "color": trajectory_df["potential"],
                        "colorscale": "Viridis",
                        "colorbar": {"title": "Potential [meV]"} if idx == 0 else None,
                        "showscale": idx == 0,
                    },
                    line={"width": 5, "color": color},
                    customdata=np.stack(
                        [
                            trajectory_df["segment_index"],
                            trajectory_df["die_id"],
                            trajectory_df["trajectory_id"],
                            trajectory_df["potential"],
                        ],
                        axis=-1,
                    ),
                    hovertemplate=(
                        "segment=%{customdata[0]}<br>"
                        "die_id=%{customdata[1]}<br>"
                        "trajectory=%{customdata[2]}<br>"
                        "potential=%{customdata[3]:.2f}<extra></extra>"
                    ),
                    name=f"Trajectory {trajectory_id}",
                )
            )

    fig.update_layout(
        margin=dict(l=0, r=0, t=20, b=0),
        scene=dict(
            xaxis_title="X [μm]",
            yaxis_title="Y [μm]",
            zaxis_title="Z [μm]",
        ),
    )
    return fig


def make_wafer_figure(wafer_df: pd.DataFrame, selected_die_ids: List[int]) -> go.Figure:
    """Construct wafer heatmap with die-level highlight."""
    max_row = int(wafer_df["row"].max()) + 1
    max_col = int(wafer_df["col"].max()) + 1
    wafer_matrix = np.full((max_row, max_col), np.nan)

    for _, row in wafer_df.iterrows():
        wafer_matrix[int(row["row"]), int(row["col"])] = row["metric"]

    fig = go.Figure(
        data=go.Heatmap(
            z=wafer_matrix,
            colorscale="Turbo",
            colorbar={"title": "Metric"},
            hovertemplate="row=%{y}<br>col=%{x}<br>metric=%{z:.2f}<extra></extra>",
        )
    )

    unique_die_ids = sorted(set(selected_die_ids or []))
    for die_id in unique_die_ids:
        selected_row = wafer_df.loc[wafer_df["die_id"] == die_id]
        if selected_row.empty:
            continue
        row = int(selected_row["row"].iloc[0])
        col = int(selected_row["col"].iloc[0])
        fig.add_shape(
            type="rect",
            x0=col - 0.5,
            x1=col + 0.5,
            y0=row - 0.5,
            y1=row + 0.5,
            line={"color": "#ff6b6b", "width": 2},
        )

    fig.update_layout(
        yaxis=dict(
            autorange="reversed",
            title="Row",
            tickmode="linear",
            dtick=1,
        ),
        xaxis=dict(title="Column", tickmode="linear", dtick=1),
        margin=dict(l=40, r=40, t=30, b=40),
    )
    return fig


def make_correlation_figure(
    waveform_df: pd.DataFrame, selected_trajectory_ids: List[int]
) -> go.Figure:
    """Render 2D scatter to inspect correlations; drives cross-highlighting."""
    if waveform_df.empty:
        return go.Figure()

    df = waveform_df.reset_index(drop=True)
    mask = df["trajectory_id"].isin(selected_trajectory_ids)
    selected_points = np.flatnonzero(mask.to_numpy()).tolist()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["amplitude"],
            y=df["metric_b"],
            mode="markers",
            marker={
                "size": 10,
                "color": df["anomaly_score"],
                "colorscale": "Plasma",
                "colorbar": {"title": "Anomaly score"},
            },
            selectedpoints=selected_points,
            selected={
                "marker": {
                    "color": "#ff6b6b",
                    "size": 14,
                }
            },
            unselected={"marker": {"opacity": 0.35}},
            customdata=np.stack(
                [
                    df["trajectory_id"],
                    df["die_id"],
                    df["anomaly_score"],
                ],
                axis=-1,
            ),
            hovertemplate=(
                "amplitude=%{x:.2f}<br>"
                "metric_b=%{y:.2f}<br>"
                "trajectory=%{customdata[0]}<br>"
                "die_id=%{customdata[1]}<br>"
                "score=%{customdata[2]:.2f}<extra></extra>"
            ),
            name="Amplitude vs Metric",
        )
    )

    fig.update_layout(
        margin=dict(l=40, r=20, t=20, b=40),
        xaxis_title="Amplitude [a.u.]",
        yaxis_title="Metric B [a.u.]",
        dragmode="lasso",
    )
    return fig


def build_info_card(
    orbit_df: pd.DataFrame,
    waveform_df: pd.DataFrame,
    selected_trajectory_ids: List[int],
) -> dbc.Card:
    """Create a metadata summary card for all active selections."""
    if not selected_trajectory_ids:
        return dbc.Card(
            dbc.Alert("No trajectory selected.", color="secondary", className="mb-0"),
            body=True,
        )

    headers = [
        "trajectory_id",
        "die_id",
        "segments",
        "path_length",
        "amplitude",
        "metric_b",
        "anomaly_score",
        "potential_min",
        "potential_max",
        "row",
        "col",
        "status",
    ]

    body_rows: List[html.Tr] = []
    for trajectory_id in selected_trajectory_ids:
        trajectory_points = orbit_df.loc[orbit_df["trajectory_id"] == trajectory_id]
        metrics_row = waveform_df.loc[waveform_df["trajectory_id"] == trajectory_id]

        if trajectory_points.empty or metrics_row.empty:
            continue

        die_id = int(metrics_row["die_id"].iloc[0])
        amplitude = round(float(metrics_row["amplitude"].iloc[0]), 3)
        metric_b = round(float(metrics_row["metric_b"].iloc[0]), 3)
        anomaly = round(float(metrics_row["anomaly_score"].iloc[0]), 3)

        potential_min = round(float(trajectory_points["potential"].min()), 3)
        potential_max = round(float(trajectory_points["potential"].max()), 3)
        segment_count = int(len(trajectory_points))

        coords = trajectory_points[["x", "y", "z"]].to_numpy()
        if len(coords) > 1:
            diffs = np.linalg.norm(np.diff(coords, axis=0), axis=1)
            path_length = round(float(diffs.sum()), 3)
        else:
            path_length = 0.0

        row = int(trajectory_points["row"].iloc[0])
        col = int(trajectory_points["col"].iloc[0])
        status = str(trajectory_points["status"].iloc[0])

        values = [
            trajectory_id,
            die_id,
            segment_count,
            path_length,
            amplitude,
            metric_b,
            anomaly,
            potential_min,
            potential_max,
            row,
            col,
            status,
        ]
        body_rows.append(
            html.Tr([html.Td(value) for value in values])
        )

    if not body_rows:
        return dbc.Card(
            dbc.Alert(
                "Selected trajectories are not present in the loaded datasets.",
                color="warning",
                className="mb-0",
            ),
            body=True,
        )

    table = dbc.Table(
        [
            html.Thead(html.Tr([html.Th(h) for h in headers])),
            html.Tbody(body_rows),
        ],
        bordered=False,
        striped=True,
        hover=True,
        size="sm",
        className="mb-0",
    )

    return dbc.Card(table, body=True, className="shadow-sm")


# ---------- Dash app factory ----------------------------------------------------------------
def create_dash_app(
    *,
    orbit_path: Optional[str | Path] = None,
    wafer_path: Optional[str | Path] = None,
    waveform_path: Optional[str | Path] = None,
    app_cls: Type[Dash] = Dash,
) -> Dash:
    """Instantiate the Dash application with linked callbacks."""
    data_bundle = build_linked_data(orbit_path, wafer_path, waveform_path)
    default_traj = data_bundle.default_selection()
    waveform_df = data_bundle.waveform
    if default_traj is not None and not waveform_df.empty and "die_id" in waveform_df.columns:
        die_series = waveform_df.loc[
            waveform_df["trajectory_id"] == default_traj, "die_id"
        ]
        if die_series.empty:
            default_die = int(waveform_df["die_id"].iloc[0])
        else:
            default_die = int(die_series.iloc[0])
    else:
        default_die = int(data_bundle.orbit["die_id"].iloc[0])

    default_traj_list = [default_traj] if default_traj is not None else []
    default_die_list = [default_die] if default_traj_list else []

    app = app_cls(
        __name__,
        external_stylesheets=[dbc.themes.SLATE],
        suppress_callback_exceptions=True,
    )

    app.layout = dbc.Container(
        [
            html.H1("Electron Trajectory & Wafer Explorer"),
            html.P(
                "Interactive linkage between 3D orbits, wafer map, and waveform spectra."
            ),
            dcc.Store(
                id="data-store",
                data={
                    "orbit": data_bundle.orbit.to_dict("records"),
                    "wafer": data_bundle.wafer.to_dict("records"),
                    "waveform": data_bundle.waveform.to_dict("records"),
                },
            ),
            dcc.Store(
                id="selection-store",
                data={"trajectory_ids": default_traj_list, "die_ids": default_die_list},
            ),
            dcc.Store(
                id="selection-buffer",
                data={"trajectory_ids": default_traj_list, "die_ids": default_die_list},
            ),
            dbc.Row(
                [
                    dbc.Col(dcc.Graph(id="orbit-graph"), width=8),
                    dbc.Col(dcc.Graph(id="wafer-graph"), width=4),
                ],
                className="gy-3",
            ),
            dbc.Row(
                [
                    dbc.Col(dcc.Graph(id="correlation-graph"), width=8),
                    dbc.Col(html.Div(id="selection-info"), width=4),
                ],
                className="gy-3",
            ),
        ],
        fluid=True,
        className="py-3",
    )

    # ---------- Selection propagation callback ---------------------------------------------
    @app.callback(
        Output("selection-store", "data"),
        Output("selection-buffer", "data"),
        Input("correlation-graph", "clickData"),
        Input("correlation-graph", "selectedData"),
        Input("wafer-graph", "clickData"),
        State("selection-store", "data"),
        State("selection-buffer", "data"),
        State("data-store", "data"),
        prevent_initial_call=True,
    )
    def update_selection(
        correlation_click: Optional[Dict[str, Any]],
        correlation_selected: Optional[Dict[str, Any]],
        wafer_click: Optional[Dict[str, Any]],
        current_selection: Optional[Dict[str, Any]],
        current_buffer: Optional[Dict[str, Any]],
        data_store: Dict[str, Any],
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """Update the global selection when the user interacts with graphs."""
        orbit_df = pd.DataFrame(data_store["orbit"])
        waveform_df = pd.DataFrame(data_store["waveform"])
        wafer_df = pd.DataFrame(data_store["wafer"])

        triggered = ctx.triggered_id or ""

        buffer_trajectories = (current_buffer or {}).get("trajectory_ids", []) or []
        buffer_dies = (current_buffer or {}).get("die_ids", []) or []
        current_trajectories = (current_selection or {}).get("trajectory_ids", []) or []
        current_dies = (current_selection or {}).get("die_ids", []) or []

        def resolve_die(tid: int) -> Optional[int]:
            series = waveform_df.loc[waveform_df["trajectory_id"] == tid, "die_id"]
            if series.empty:
                series = orbit_df.loc[orbit_df["trajectory_id"] == tid, "die_id"]
            if series.empty:
                return None
            return int(series.iloc[0])

        current_pairs: "OrderedDict[int, int]" = OrderedDict()
        for tid, did in zip(buffer_trajectories, buffer_dies):
            if tid is None:
                continue
            tid_int = int(tid)
            die_val = int(did) if did is not None else resolve_die(tid_int)
            if die_val is None or tid_int in current_pairs:
                continue
            current_pairs[tid_int] = die_val
        if not current_pairs:
            for tid, did in zip(current_trajectories, current_dies):
                if tid is None:
                    continue
                tid_int = int(tid)
                die_val = int(did) if did is not None else resolve_die(tid_int)
                if die_val is None or tid_int in current_pairs:
                    continue
                current_pairs[tid_int] = die_val

        if triggered.endswith(".clickData") and not correlation_click:
            # Background click without Ctrl clears selection
            cleared = {"trajectory_ids": [], "die_ids": []}
            return cleared, cleared

        def to_store(pairs: "OrderedDict[int, int]") -> Dict[str, Any]:
            return {
                "trajectory_ids": list(pairs.keys()),
                "die_ids": list(pairs.values()),
            }

        if correlation_selected:
            points = correlation_selected.get("points") or []
            pairs = current_pairs.copy()
            if points:
                for point in points:
                    customdata = point.get("customdata")
                    if customdata is None:
                        continue
                    trajectory_id = int(customdata[0])
                    die_id = int(customdata[1])
                    if trajectory_id not in pairs:
                        pairs[trajectory_id] = die_id
                store = to_store(pairs)
                return store, store
            cleared = {"trajectory_ids": [], "die_ids": []}
            return cleared, cleared

        if correlation_click and correlation_click.get("points"):
            point = correlation_click["points"][0]
            customdata = point.get("customdata")
            if customdata is not None:
                trajectory_id = int(customdata[0])
                die_id = int(customdata[1])
                pairs = current_pairs.copy()
                if trajectory_id not in pairs:
                    pairs[trajectory_id] = die_id
                store = to_store(pairs)
                return store, store
            x_val = point.get("x")
            y_val = point.get("y")
            if x_val is not None and y_val is not None and not waveform_df.empty:
                diffs = (
                    (waveform_df["amplitude"] - float(x_val)).abs()
                    + (waveform_df["metric_b"] - float(y_val)).abs()
                )
                idx = int(diffs.idxmin())
                row_match = waveform_df.loc[idx]
                trajectory_id = int(row_match["trajectory_id"])
                die_id = int(row_match["die_id"])
                pairs = current_pairs.copy()
                if trajectory_id not in pairs:
                    pairs[trajectory_id] = die_id
                store = to_store(pairs)
                return store, store

        if wafer_click and wafer_click.get("points"):
            point = wafer_click["points"][0]
            col = int(point["x"])
            row = int(point["y"])
            wafer_match = wafer_df.loc[(wafer_df["row"] == row) & (wafer_df["col"] == col)]
            if not wafer_match.empty:
                die_id = int(wafer_match["die_id"].iloc[0])
                trajectory_match = waveform_df.loc[waveform_df["die_id"] == die_id]
                trajectories = (
                    trajectory_match["trajectory_id"].astype(int).unique().tolist()
                    if not trajectory_match.empty
                    else []
                )
                if not trajectories:
                    matched_tid = None
                    for tid in current_pairs.keys():
                        if resolve_die(tid) == die_id:
                            matched_tid = tid
                            break
                    if matched_tid is not None:
                        trajectories = [matched_tid]
                if trajectories:
                    pairs = current_pairs.copy()
                    for tid in trajectories:
                        if tid not in pairs:
                            resolved_die = resolve_die(tid)
                            if resolved_die is not None:
                                pairs[tid] = resolved_die
                    store = to_store(pairs)
                    return store, store

        store = to_store(current_pairs)
        return store, store

    # ---------- Figure updating callback ----------------------------------------------------
    @app.callback(
        Output("orbit-graph", "figure"),
        Output("wafer-graph", "figure"),
        Output("correlation-graph", "figure"),
        Output("selection-info", "children"),
        Input("selection-store", "data"),
        State("data-store", "data"),
    )
    def refresh_figures(
        selection: Dict[str, Any],
        data_store: Dict[str, Any],
    ) -> tuple[go.Figure, go.Figure, go.Figure, dbc.Card]:
        """Re-render all figures based on the current selection."""
        orbit_df = pd.DataFrame(data_store["orbit"])
        wafer_df = pd.DataFrame(data_store["wafer"])
        waveform_df = pd.DataFrame(data_store["waveform"])
        trajectory_ids = (
            [int(tid) for tid in selection.get("trajectory_ids", [])]
            if selection
            else []
        )
        die_ids = (
            [int(did) for did in selection.get("die_ids", []) if did is not None]
            if selection
            else []
        )

        if not trajectory_ids:
            if not waveform_df.empty:
                trajectory_ids = [int(waveform_df["trajectory_id"].iloc[0])]
                die_ids = [int(waveform_df["die_id"].iloc[0])]
            elif not orbit_df.empty:
                traj_series = orbit_df["trajectory_id"].dropna()
                if not traj_series.empty:
                    trajectory_ids = [int(traj_series.iloc[0])]
                    die_lookup = waveform_df.loc[
                        waveform_df["trajectory_id"] == trajectory_ids[0], "die_id"
                    ]
                    if not die_lookup.empty:
                        die_ids = [int(die_lookup.iloc[0])]
                    else:
                        die_ids = [int(orbit_df.loc[orbit_df["trajectory_id"] == trajectory_ids[0], "die_id"].iloc[0])]
            else:
                trajectory_ids = []
                die_ids = []

        def resolve_die(tid: int) -> Optional[int]:
            series = waveform_df.loc[waveform_df["trajectory_id"] == tid, "die_id"]
            if series.empty:
                series = orbit_df.loc[orbit_df["trajectory_id"] == tid, "die_id"]
            if series.empty:
                return None
            return int(series.iloc[0])

        resolved_pairs: List[tuple[int, int]] = []
        seen: set[int] = set()
        for tid in trajectory_ids:
            if tid in seen:
                continue
            seen.add(tid)
            did = resolve_die(tid)
            if did is None:
                continue
            resolved_pairs.append((tid, did))

        if not resolved_pairs and not waveform_df.empty:
            tid = int(waveform_df["trajectory_id"].iloc[0])
            did = int(waveform_df["die_id"].iloc[0])
            resolved_pairs.append((tid, did))

        trajectory_ids = [tid for tid, _ in resolved_pairs]
        die_ids = [did for _, did in resolved_pairs]

        return (
            make_orbit_figure(orbit_df, trajectory_ids),
            make_wafer_figure(wafer_df, die_ids),
            make_correlation_figure(waveform_df, trajectory_ids),
            build_info_card(orbit_df, waveform_df, trajectory_ids),
        )

    return app


# ---------- Notebook convenience ------------------------------------------------------------
def notebook_entry(
    orbit_path: Optional[str | Path] = None,
    wafer_path: Optional[str | Path] = None,
    waveform_path: Optional[str | Path] = None,
    *,
    mode: str = "inline",
    port: int = 8050,
) -> "jupyter_dash.JupyterDash":
    """Launch the dashboard within Jupyter using JupyterDash."""
    from jupyter_dash import JupyterDash

    dash_app = create_dash_app(
        orbit_path=orbit_path,
        wafer_path=wafer_path,
        waveform_path=waveform_path,
        app_cls=JupyterDash,
    )
    dash_app.run(jupyter_mode=mode, port=port)
    return dash_app


# ---------- CLI ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch the electron trajectory & wafer dashboard."
    )
    parser.add_argument("--orbit", type=Path, default=None, help="Path to orbit CSV.")
    parser.add_argument("--wafer", type=Path, default=None, help="Path to wafer CSV.")
    parser.add_argument(
        "--waveform", type=Path, default=None, help="Path to waveform CSV."
    )
    parser.add_argument(
        "--port", type=int, default=8050, help="Host port for the Dash server."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run Dash with debug mode (enable reloader and hot reload).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = create_dash_app(
        orbit_path=args.orbit,
        wafer_path=args.wafer,
        waveform_path=args.waveform,
    )
    app.run(
        debug=args.debug,
        port=args.port,
        use_reloader=args.debug,
    )


if __name__ == "__main__":
    main()
