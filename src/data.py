
"""
Data loading and preprocessing utilities.
"""

import numpy as np
import polars as pl
import pandas as pd


META_COLS = ["date_id", "time_id", "symbol_id", "weight"]
TARGET_COL = "responder_6"


def make_purged_splits(min_d, max_d, n_splits=4, purge_days=5, max_train_days=None):
    total_span = (max_d - min_d + 1)
    chunk = total_span // (n_splits + 1)
    splits = []
    for i in range(1, n_splits + 1):
        train_end = min_d + (i * chunk) - purge_days - 1
        train_start = min_d if max_train_days is None else max(min_d, train_end - max_train_days + 1)
        val_start = min_d + (i * chunk)
        val_end = val_start + chunk - 1 if i < n_splits else max_d
        splits.append({
            "fold": i,
            "train_range": (int(train_start), int(train_end)),
            "val_range": (int(val_start), int(val_end))
        })
    return splits


def compute_training_stats_batched(lazy_df, train_range, cont_cols, batch_size=20):
    t_min, t_max = train_range
    train_slice = lazy_df.filter((pl.col("date_id") >= t_min) & (pl.col("date_id") <= t_max))
    means, stds = {}, {}
    for i in range(0, len(cont_cols), batch_size):
        batch = cont_cols[i : i + batch_size]
        exprs = []
        for c in batch:
            exprs.append(pl.col(c).cast(pl.Float64).mean().alias(f"{c}__mean"))
            exprs.append(pl.col(c).cast(pl.Float64).std().alias(f"{c}__std"))
        stats_row = train_slice.select(exprs).collect(streaming=True).to_dicts()[0]
        for c in batch:
            m = stats_row[f"{c}__mean"]
            s = stats_row[f"{c}__std"]
            means[c] = float(m) if (m is not None and m == m) else 0.0
            stds[c] = float(s) if (s is not None and s == s and s > 1e-8) else 1.0
        print(f"  Computed stats for features {i+1}..{min(i+batch_size, len(cont_cols))}/{len(cont_cols)}")
    return means, stds


def build_standardized_zero_imputed_lazy(
    lazy_df, date_range, cont_cols, disc_cols, means, stds,
    clip_bound=10.0, filter_positive_weights=True
):
    d_min, d_max = date_range
    sub_lazy = lazy_df.filter((pl.col("date_id") >= d_min) & (pl.col("date_id") <= d_max))
    if filter_positive_weights:
        sub_lazy = sub_lazy.filter(pl.col("weight") > 0)

    cont_exprs = [
        (
            ((pl.col(c).cast(pl.Float32) - means[c]) / stds[c])
            .clip(-clip_bound, clip_bound)
            .fill_nan(0.0)
            .fill_null(0.0)
            .cast(pl.Float32)
            .alias(c)
        )
        for c in cont_cols
    ]
    disc_exprs = [pl.col(c).fill_null(0).cast(pl.Float32).alias(c) for c in disc_cols]
    meta_exprs = [
        pl.col("date_id").cast(pl.Int16),
        pl.col("time_id").cast(pl.Int16),
        pl.col("symbol_id").fill_null(-1).cast(pl.Int16),
        pl.col("weight").fill_null(0.0).cast(pl.Float32),
        pl.col(TARGET_COL).fill_null(0.0).cast(pl.Float32)
    ]
    selected_cols = META_COLS + [TARGET_COL] + cont_cols + disc_cols
    return sub_lazy.with_columns(cont_exprs + disc_exprs + meta_exprs).select(selected_cols)
