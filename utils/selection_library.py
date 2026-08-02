"""Selection persistence and tabular statistics export."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


class SelectionLibrary:
    """Serialize selection masks and histogram ROIs without losing colour types."""

    VERSION = "1.1"

    @staticmethod
    def _encode_color(color: Any) -> dict | None:
        if color is None:
            return None
        if isinstance(color, str):
            return {"format": "string", "value": color}
        values = np.asarray(color).reshape(-1)
        if values.size not in (3, 4):
            raise ValueError("Selection colour must be a string, RGB, or RGBA value")
        return {
            "format": "rgba" if values.size == 4 else "rgb",
            "value": [float(value) for value in values],
        }

    @staticmethod
    def _decode_color(payload: Any) -> Any:
        if payload is None:
            return None
        if isinstance(payload, str):
            return payload
        if isinstance(payload, dict):
            value = payload.get("value")
            if payload.get("format") == "string":
                return str(value)
            if payload.get("format") in {"rgb", "rgba"}:
                return tuple(float(component) for component in value)
            raise ValueError(f"Unsupported colour format: {payload.get('format')}")
        # Backward compatibility with version 1.0 numeric lists.  A legacy
        # hexadecimal string must remain a string rather than becoming a tuple
        # of characters.
        if isinstance(payload, list):
            if all(isinstance(component, (int, float)) for component in payload):
                return tuple(float(component) for component in payload)
            return "".join(str(component) for component in payload)
        return payload

    @staticmethod
    def save_selections(selections: list, filepath: str | Path) -> str:
        data = {
            "version": SelectionLibrary.VERSION,
            "created": datetime.now(timezone.utc).isoformat(),
            "num_selections": len(selections),
            "selections": [],
        }
        for selection in selections:
            spatial_mask = getattr(selection, "spatial_mask", None)
            histogram_roi = getattr(selection, "histogram_roi", None)
            data["selections"].append(
                {
                    "name": str(selection.name),
                    "cluster_id": (
                        None
                        if getattr(selection, "cluster_id", None) is None
                        else int(selection.cluster_id)
                    ),
                    "visible": bool(getattr(selection, "visible", True)),
                    "spatial_mask": (
                        None
                        if spatial_mask is None
                        else np.asarray(spatial_mask, dtype=bool).tolist()
                    ),
                    "histogram_roi": (
                        None
                        if histogram_roi is None
                        else np.asarray(histogram_roi, dtype=float).tolist()
                    ),
                    "color": SelectionLibrary._encode_color(
                        getattr(selection, "color", None)
                    ),
                }
            )

        output = Path(filepath)
        if output.suffix.lower() != ".bits":
            output = output.with_suffix(".bits")
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, allow_nan=False)
        return str(output)

    @staticmethod
    def load_selections(filepath: str | Path) -> list[dict]:
        source = Path(filepath)
        if not source.exists():
            raise FileNotFoundError(f"Not found: {source}")
        with source.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if "selections" not in data or not isinstance(data["selections"], list):
            raise ValueError("Invalid .bits file: missing selections list")

        selections: list[dict] = []
        for item in data["selections"]:
            mask = item.get("spatial_mask")
            roi = item.get("histogram_roi")
            selections.append(
                {
                    "name": str(item["name"]),
                    "spatial_mask": (
                        None if mask is None else np.asarray(mask, dtype=bool)
                    ),
                    "histogram_roi": (
                        None if roi is None else np.asarray(roi, dtype=float)
                    ),
                    "cluster_id": item.get("cluster_id"),
                    "color": SelectionLibrary._decode_color(item.get("color")),
                    "visible": bool(item.get("visible", True)),
                }
            )
        return selections

    @staticmethod
    def _statistics_rows(selections, neutron_data, xray_data):
        neutron = np.asarray(neutron_data)
        xray = np.asarray(xray_data)
        if neutron.shape != xray.shape:
            raise ValueError("Neutron and X-ray arrays must have identical shapes")

        total_pixels = neutron.size
        rows = []
        skipped = 0
        for selection in selections:
            mask = getattr(selection, "spatial_mask", None)
            if mask is None:
                continue
            mask = np.asarray(mask, dtype=bool)
            if mask.shape != neutron.shape:
                skipped += 1
                continue
            count = int(np.count_nonzero(mask))
            if count == 0:
                continue
            neutron_values = neutron[mask]
            xray_values = xray[mask]
            rows.append(
                {
                    "Selection": selection.name,
                    "Cluster_ID": getattr(selection, "cluster_id", None),
                    "Pixels": count,
                    "Volume_%": 100.0 * count / total_pixels,
                    "Neutron_Mean": float(np.mean(neutron_values)),
                    "Neutron_Std": float(np.std(neutron_values)),
                    "Neutron_Min": float(np.min(neutron_values)),
                    "Neutron_Max": float(np.max(neutron_values)),
                    "Xray_Mean": float(np.mean(xray_values)),
                    "Xray_Std": float(np.std(xray_values)),
                    "Xray_Min": float(np.min(xray_values)),
                    "Xray_Max": float(np.max(xray_values)),
                }
            )
        return rows, skipped

    @staticmethod
    def export_statistics_csv(
        selections, neutron_data, xray_data, filepath: str | Path
    ) -> tuple[str, int]:
        rows, skipped = SelectionLibrary._statistics_rows(
            selections, neutron_data, xray_data
        )
        output = Path(filepath)
        if output.suffix.lower() != ".csv":
            output = output.with_suffix(".csv")
        output.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "Selection",
            "Cluster_ID",
            "Pixels",
            "Volume_%",
            "Neutron_Mean",
            "Neutron_Std",
            "Neutron_Min",
            "Neutron_Max",
            "Xray_Mean",
            "Xray_Std",
            "Xray_Min",
            "Xray_Max",
        ]
        with output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return str(output), skipped

    @staticmethod
    def export_statistics_excel(
        selections, neutron_data, xray_data, filepath: str | Path
    ) -> tuple[str, int]:
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("Install pandas and openpyxl for Excel export") from exc

        rows, skipped = SelectionLibrary._statistics_rows(
            selections, neutron_data, xray_data
        )
        output = Path(filepath)
        if output.suffix.lower() not in {".xlsx", ".xls"}:
            output = output.with_suffix(".xlsx")
        output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_excel(output, index=False, engine="openpyxl")
        return str(output), skipped
