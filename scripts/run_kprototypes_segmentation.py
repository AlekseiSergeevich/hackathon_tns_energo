#!/usr/bin/env python3
"""
K-Prototypes сегментация по готовой таблице признаков (план 2.1).

Читает output/kprototypes_features.csv: числовые колонки стандартизуются,
категориальные — из фиксированного списка (как в build_kprototypes_features.py).

По умолчанию: обучение на случайной подвыборке (ускорение), кластер для всех ЛС
через predict. Итог: метки, профили сегментов, JSON с метаданными запуска.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from kmodes.kprototypes import KPrototypes
from sklearn.preprocessing import StandardScaler

# Совпадает с fill_missing_for_kprototypes в build_kprototypes_features.py
CATEGORICAL_COLS: tuple[str, ...] = (
    "has_phone",
    "has_ebill",
    "is_gasified",
    "has_benefits",
    "not_residing",
    "remote_disconnect_possible",
    "room_type",
    "meter_installation",
    "settlement_type",
    "stage_reached",
    "dominant_payment_method",
    "ever_restricted",
    "ever_in_court",
)

ID_AND_META: frozenset[str] = frozenset({"ls", "snapshot_month", "asof_date"})


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj) if isinstance(obj, np.floating) else int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (datetime, date, pd.Timestamp)):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    return obj


def build_feature_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Порядок: сначала все числовые, затем категориальные (требование kmodes)."""
    d = df.copy()
    all_cols = [c for c in d.columns if c not in ID_AND_META]
    present_cat = [c for c in CATEGORICAL_COLS if c in all_cols]
    num_cols = [c for c in all_cols if c not in set(present_cat)]
    for c in num_cols:
        d[c] = pd.to_numeric(d[c], errors="coerce")
        med = d[c].median()
        d[c] = d[c].fillna(med if pd.notna(med) else 0.0)
    for c in present_cat:
        d[c] = d[c].astype("string").fillna("Unknown").map(lambda x: str(x).strip() or "Unknown")
    ordered = num_cols + present_cat
    return d[ordered], num_cols, present_cat


def frame_to_kproto_matrix(
    xdf: pd.DataFrame, num_cols: list[str], cat_cols: list[str]
) -> tuple[np.ndarray, list[int]]:
    """Матрица (n, n_num + n_cat), категориальные индексы с n_num."""
    n, nnum = xdf.shape[0], len(num_cols)
    if not cat_cols:
        raise SystemExit("Нет категориальных колонок — для чисто числовых данных используйте KMeans.")
    x = np.empty((n, nnum + len(cat_cols)), dtype=object)
    num_part = xdf[num_cols].to_numpy(dtype=np.float64)
    for j in range(nnum):
        x[:, j] = num_part[:, j]
    for j, c in enumerate(cat_cols):
        x[:, nnum + j] = xdf[c].to_numpy()
    cat_idx = list(range(nnum, nnum + len(cat_cols)))
    return x, cat_idx


def run(
    input_csv: Path,
    out_labels: Path,
    out_profile: Path,
    out_meta: Path,
    n_clusters: int,
    fit_sample: int | None,
    random_state: int,
    n_init: int,
    max_iter: int,
    n_jobs: int,
) -> None:
    df = pd.read_csv(input_csv)
    if "ls" not in df.columns:
        raise SystemExit("В CSV нет колонки ls")

    feats, num_cols, cat_cols = build_feature_frame(df)
    if not num_cols:
        raise SystemExit("Нет числовых колонок после исключения id/meta")

    x_full, cat_idx = frame_to_kproto_matrix(feats, num_cols, cat_cols)
    nnum = len(num_cols)
    x_num = x_full[:, :nnum].astype(np.float64)
    scaler = StandardScaler()
    scaler.fit(x_num)
    x_full[:, :nnum] = scaler.transform(x_num)

    rng = np.random.default_rng(random_state)
    n = x_full.shape[0]
    if fit_sample is not None and fit_sample < n:
        idx = rng.choice(n, size=fit_sample, replace=False)
        x_fit = x_full[idx]
    else:
        x_fit = x_full

    kproto = KPrototypes(
        n_clusters=n_clusters,
        max_iter=max_iter,
        init="Cao",
        n_init=n_init,
        n_jobs=n_jobs,
        random_state=random_state,
        verbose=0,
    )
    kproto.fit(x_fit, categorical=cat_idx)
    labels = kproto.predict(x_full, categorical=cat_idx)

    out = pd.DataFrame({"ls": df["ls"].values, "segment": labels})
    out_labels.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_labels, index=False)

    # Профили: доля, медианы ключевых числовых, моды по stage_reached
    prof_rows = []
    for seg in range(n_clusters):
        m = labels == seg
        row: dict[str, Any] = {
            "segment": seg,
            "n": int(m.sum()),
            "share": float(m.mean()),
        }
        for key in (
            "DZ_current",
            "payment_ratio_12m",
            "log1p_DZ_current",
            "n_measures_12m",
        ):
            if key in df.columns:
                s = pd.to_numeric(df.loc[m, key], errors="coerce")
                row[f"median_{key}"] = float(s.median()) if s.notna().any() else None
        if "stage_reached" in df.columns:
            mode = df.loc[m, "stage_reached"].astype(str).mode()
            row["mode_stage_reached"] = str(mode.iloc[0]) if len(mode) else None
        prof_rows.append(row)
    profile_df = pd.DataFrame(prof_rows)
    profile_df.to_csv(out_profile, index=False)

    cost = float(kproto.cost_) if hasattr(kproto, "cost_") else None
    gamma = float(kproto.gamma) if kproto.gamma is not None else None
    meta = {
        "input_csv": str(input_csv.resolve()),
        "n_rows": n,
        "n_clusters": n_clusters,
        "n_numeric": len(num_cols),
        "n_categorical": len(cat_cols),
        "numeric_columns": num_cols,
        "categorical_columns": cat_cols,
        "fit_sample_size": int(x_fit.shape[0]),
        "cost_on_fit": cost,
        "gamma": gamma,
        "n_iter": int(kproto.n_iter_) if hasattr(kproto, "n_iter_") else None,
        "random_state": random_state,
    }
    out_meta.write_text(json.dumps(_to_jsonable(meta), ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(_to_jsonable(meta), ensure_ascii=False, indent=2), file=sys.stderr)
    print(f"Labels: {out_labels}", file=sys.stderr)
    print(f"Profile: {out_profile}", file=sys.stderr)
    print(f"Meta: {out_meta}", file=sys.stderr)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="K-Prototypes по kprototypes_features.csv")
    p.add_argument(
        "--input-csv",
        type=Path,
        default=Path("output/kprototypes_features.csv"),
    )
    p.add_argument("--n-clusters", type=int, default=6)
    p.add_argument(
        "--fit-sample",
        type=int,
        default=50_000,
        help="Число строк для fit (по умолчанию 50000; меньше — быстрее). 0 = вся выборка.",
    )
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--n-init", type=int, default=1)
    p.add_argument("--max-iter", type=int, default=30)
    p.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Параллелизм n_init в KPrototypes (-1 = все CPU; в sandbox оставьте 1).",
    )
    p.add_argument("--out-labels", type=Path, default=Path("output/kprototypes_labels.csv"))
    p.add_argument("--out-profile", type=Path, default=Path("output/kprototypes_segment_profile.csv"))
    p.add_argument("--out-meta", type=Path, default=Path("output/kprototypes_run_meta.json"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    fit_sample = None if args.fit_sample == 0 else args.fit_sample
    run(
        input_csv=args.input_csv,
        out_labels=args.out_labels,
        out_profile=args.out_profile,
        out_meta=args.out_meta,
        n_clusters=args.n_clusters,
        fit_sample=fit_sample,
        random_state=args.random_state,
        n_init=args.n_init,
        max_iter=args.max_iter,
        n_jobs=args.n_jobs,
    )


if __name__ == "__main__":
    main()
