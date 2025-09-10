#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
vague.py — Viral waves detection with hysteresis (K/H), optional normalization and temporal redistribution.
Outputs a per-post diagnostic table.

Privacy note: avoid committing raw data. Use --anonymize to hash identifiers when exporting.
"""

from __future__ import annotations
import argparse
import hashlib
from pathlib import Path
import pandas as pd
import numpy as np

# --------------------------- Config defaults ---------------------------
COL_DATE   = "Post Created"
COL_CORPUS = "Corpus"
COL_POSTID = "Post_ID"

WINDOW_DEFAULT   = 7
K_DEFAULT        = 1.0   # upper threshold
H_DEFAULT        = 0.8   # lower threshold (H < K)
MAX_LOCAL_PEAK   = True
NB_POSTS_WINDOW  = 3

NORMALIZE_ENG_DEFAULT = False
REDISTRIBUTE_DEFAULT  = False
KERNEL_DEFAULT        = [0.5, 0.3, 0.15, 0.05]  # J0..J3, will be normalized to sum=1

# --------------------------- Helpers ---------------------------
def _hash_series(s: pd.Series) -> pd.Series:
    return s.astype(str).apply(lambda x: hashlib.sha256(x.encode("utf-8")).hexdigest())

def read_base(path: Path, col_date: str, col_corpus: str, col_postid: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", encoding="utf-8-sig", low_memory=False)
    df.columns = df.columns.str.strip()
    missing = [c for c in (col_date, col_corpus, col_postid) if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    df[col_date] = pd.to_datetime(df[col_date], dayfirst=True, errors="coerce")
    df = df.dropna(subset=[col_date, col_corpus])
    df["date"] = df[col_date].dt.floor("D")
    return df

def compute_engagement(df: pd.DataFrame, normalize: bool = False) -> pd.DataFrame:
    comps_all = [c for c in ["Likes", "Comments", "Shares", "Views", "Reactions"] if c in df.columns]
    if not normalize:
        if "Total Interactions" in df.columns:
            df["eng"] = pd.to_numeric(df["Total Interactions"], errors="coerce").fillna(0)
        else:
            if not comps_all:
                raise ValueError("No engagement fields available.")
            df["eng"] = df[comps_all].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1)
        return df

    base_cols = [c for c in ["Likes", "Comments", "Shares"] if c in df.columns]
    if not base_cols:
        raise ValueError("Normalization requires at least Likes/Comments/Shares.")
    for c in base_cols:
        v = pd.to_numeric(df[c], errors="coerce").fillna(0)
        m = v.mean()
        df[f"{c}_norm"] = 0.0 if m <= 0 else v / m
    df["eng"] = df[[f"{c}_norm" for c in base_cols]].sum(axis=1)
    return df

def redistribute_daily(series_daily: pd.Series, kernel: list[float]) -> pd.Series:
    k = [float(x) for x in kernel]
    s = sum(k)
    k = [x / s for x in k] if s != 1.0 else k
    out = pd.Series(0.0, index=series_daily.index)
    for lag, w in enumerate(k):
        out = out.add(series_daily.shift(lag).fillna(0.0) * w, fill_value=0.0)
    return out

def compute_rolling_and_thresholds(q: pd.Series, window: int, K: float, H: float):
    roll = q.rolling(window, min_periods=1)
    mean = roll.mean()
    std  = roll.std(ddof=0)
    up   = mean + K * std
    low  = mean + H * std
    return mean, std, up, low

def hysteresis_episodes(q: pd.Series, seuil_K: pd.Series, seuil_H: pd.Series) -> pd.Series:
    in_wave = False
    ep = 0
    out = np.zeros(len(q), dtype=int)
    for i, (val, up, low) in enumerate(zip(q.values, seuil_K.values, seuil_H.values)):
        if not in_wave and val > up:
            in_wave = True
            ep += 1
            out[i] = ep
        elif in_wave and val > low:
            out[i] = ep
        else:
            in_wave = False
            out[i] = 0
    return pd.Series(out, index=q.index, dtype=int)

def mark_local_peaks(q: pd.Series, seuil_K: pd.Series, max_local: bool = True) -> pd.Series:
    if max_local:
        prev = q.shift(1)
        nxt  = q.shift(-1)
        is_max = (q > prev) & (q > nxt)
    else:
        is_max = pd.Series(True, index=q.index)
    return ((q > seuil_K) & is_max).astype(bool)

def build_post_level(
    df_corpus: pd.DataFrame,
    nom_corpus: str,
    window: int,
    K: float,
    H: float,
    max_local: bool,
    redistribute: bool,
    kernel: list[float],
) -> pd.DataFrame:
    daily = df_corpus.groupby("date", as_index=True)["eng"].sum().sort_index()
    idx = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(idx, fill_value=0.0)
    daily.index.name = "date"

    q_eff = redistribute_daily(daily, kernel) if redistribute else daily.copy()
    mean, std, up, low = compute_rolling_and_thresholds(q_eff, window, K, H)
    is_peak_day = mark_local_peaks(q_eff, up, max_local=max_local)
    episodes = hysteresis_episodes(q_eff, up, low)

    df_days = pd.DataFrame({
        "date": q_eff.index,
        "valeur": q_eff.values,
        "moyenne": mean.values,
        "ecart_type": std.values,
        "seuil_sup": up.values,
        "seuil_inf": low.values,
        "est_pic": is_peak_day.values,
        "episode": episodes.values,
    })

    df_posts = df_corpus.copy().sort_values(["date", COL_POSTID]).reset_index(drop=True)
    df_posts["corpus"] = nom_corpus
    df_posts = df_posts.merge(df_days, on="date", how="left")

    prev = df_posts["est_pic"].shift(1).fillna(False).astype(bool)
    nxt  = df_posts["est_pic"].shift(-1).fillna(False).astype(bool)
    df_posts["est_pic_precedent"] = prev
    df_posts["est_pic_suivant"]   = nxt

    df_posts["est_pic_avant"] = (
        df_posts["est_pic"].rolling(window=NB_POSTS_WINDOW, min_periods=1)
        .max().shift(1).fillna(0).astype(int).astype(bool)
    )
    inv = df_posts["est_pic"][::-1]
    df_posts["est_pic_apres"] = (
        inv.rolling(window=NB_POSTS_WINDOW, min_periods=1)
        .max().shift(1).fillna(0).astype(int)
    )[::-1].astype(bool)

    peaks_idx = df_posts.index[df_posts["est_pic"]].to_numpy()

    def dmin(i: int, pics: np.ndarray) -> float:
        if pics.size == 0:
            return np.nan
        return int(np.min(np.abs(i - pics)))

    df_posts["distance_au_pic"] = df_posts.index.to_series().apply(lambda i: dmin(i, peaks_idx))

    cols = [
        "corpus", COL_POSTID, "date", "eng",
        "valeur", "moyenne", "ecart_type", "seuil_sup", "seuil_inf", "est_pic",
        "est_pic_precedent", "est_pic_suivant", "est_pic_avant", "est_pic_apres",
        "distance_au_pic", "episode",
    ]
    return df_posts[cols]

# --------------------------- CLI ---------------------------
def main():
    p = argparse.ArgumentParser(description="Detect viral waves with hysteresis.")
    p.add_argument("--input", required=True, type=Path, help="Input CSV (semicolon-separated, utf-8-sig).")
    p.add_argument("--output", required=True, type=Path, help="Output CSV path (per-post diagnostics).")
    p.add_argument("--window", type=int, default=WINDOW_DEFAULT)
    p.add_argument("--K", type=float, default=K_DEFAULT)
    p.add_argument("--H", type=float, default=H_DEFAULT)
    p.add_argument("--no-maxlocal", action="store_true", help="Do not require local maxima for peaks.")
    p.add_argument("--normalize-eng", action="store_true", default=NORMALIZE_ENG_DEFAULT)
    p.add_argument("--redistribute", action="store_true", default=REDISTRIBUTE_DEFAULT)
    p.add_argument("--kernel", type=float, nargs="+", default=KERNEL_DEFAULT, help="Redistribution kernel J0..Jn.")
    p.add_argument("--anonymize", action="store_true", help="Hash identifiers (Post_ID) before saving.")
    p.add_argument("--drop-ids", action="store_true", help="Drop potential PII columns if present (User/Page/URL).")
    args = p.parse_args()

    df = read_base(args.input, COL_DATE, COL_CORPUS, COL_POSTID)
    df = compute_engagement(df, normalize=args.normalize_eng)

    outputs = []
    for corpus, df_c in df.groupby(COL_CORPUS):
        out = build_post_level(
            df_c, corpus, args.window, args.K, args.H,
            max_local=(not args.no_maxlocal),
            redistribute=args.redistribute,
            kernel=args.kernel,
        )
        outputs.append(out)

    result = pd.concat(outputs, ignore_index=True)

    if args.anonymize and COL_POSTID in result.columns:
        result[COL_POSTID] = _hash_series(result[COL_POSTID])

    if args.drop_ids:
        for c in ["Facebook Id", "User Name", "Page Name", "URL", "Final Link", "Link"]:
            if c in result.columns:
                result = result.drop(columns=[c])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False, encoding="utf-8")

if __name__ == "__main__":
    main()
