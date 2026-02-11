#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def _load_summary(path: Path) -> dict[tuple[str, str, str, str, str], dict[str, str]]:
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    data: dict[tuple[str, str, str, str, str], dict[str, str]] = {}
    for row in rows:
        key = (
            row.get("dataset", ""),
            row.get("method", ""),
            row.get("k_nn", ""),
            row.get("lambda_reg", ""),
            row.get("subsample", ""),
        )
        data[key] = row
    return data


def _to_float(value: str) -> float | None:
    if value is None:
        return None
    value = value.strip()
    if value == "":
        return None
    try:
        return float(value)
    except Exception:
        return None


def main() -> None:
    p = argparse.ArgumentParser(description="Compare validation summary.csv against a baseline.")
    p.add_argument("--baseline", required=True, type=Path)
    p.add_argument("--current", required=True, type=Path)
    p.add_argument("--rtol", type=float, default=0.15)
    p.add_argument("--atol", type=float, default=1e-8)
    args = p.parse_args()

    baseline = _load_summary(args.baseline)
    current = _load_summary(args.current)

    columns = ["rms", "p95", "max", "bim_mfs_diff_rms"]
    errors: list[str] = []

    missing = [k for k in baseline.keys() if k not in current]
    if missing:
        errors.append(f"Missing {len(missing)} baseline rows in current summary.")

    for key, brow in baseline.items():
        crow = current.get(key)
        if crow is None:
            continue
        for col in columns:
            bval = _to_float(brow.get(col, ""))
            cval = _to_float(crow.get(col, ""))
            if bval is None or cval is None:
                continue
            if not (math.isfinite(bval) and math.isfinite(cval)):
                errors.append(f"Non-finite values for {key} column {col}.")
                continue
            tol = args.atol + args.rtol * abs(bval)
            if abs(cval - bval) > tol:
                errors.append(
                    f"{key} {col} drift: baseline={bval:.3e}, current={cval:.3e}, tol={tol:.3e}"
                )

    if errors:
        print("[FAIL] Validation baseline comparison failed:")
        for err in errors:
            print(" -", err)
        raise SystemExit(1)

    print("[OK] Validation baseline comparison passed.")


if __name__ == "__main__":
    main()
