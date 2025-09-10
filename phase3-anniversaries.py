#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
phase3-anniversaries.py — Anniversary & reactivation analysis over viral-wave diagnostics.

Inputs:
  - A CSV produced by Phase 1 (waves/peaks diagnostics), with at least:
      date, valeur (daily intensity), and optionally: seuil_sup or seuil, est_pic, corpus/Corpus
  - Optional CSV mapping 'corpus,anchor_date' (YYYY-MM-DD) for anniversary anchors.

Privacy & Repo Hygiene:
  - No raw datasets committed to GitHub.
  - No secrets or local absolute paths.
  - Outputs only aggregate stats and plots.

Examples:
  # Single corpus + anchor via CLI, 7-day window
  python scripts/phase3-anniversaries.py \
    --input outputs/posts_diagnostics_vagues.csv \
    --output-dir outputs/anniversaries \
    --corpus "Marielle Franco" \
    --anchor 2018-03-14 \
    --window 7

  # Multiple corpora with anchors.csv (columns: corpus,anchor_date)
  python scripts/phase3-anniversaries.py \
    --input outputs/posts_diagnostics_vagues.csv \
    --output-dir outputs/anniversaries \
    --anchors-csv config/anchors.csv \
    --window 7 --top-peaks 5
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------------------- IO helpers --------------------
def read_csv_smart(path: Path) -> pd.DataFrame:
    """Try to sniff the separator; fall back to ',' then ';'."""
    df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8", low_memory=False)
    if df.shape[1] == 1:
        df = pd.read_csv(path, sep=",", engine="python", encoding="utf-8", low_memory=False)
    if df.shape[1] == 1:
        df = pd.read_csv(path, sep=";", engine="python", encoding="utf-8", low_memory=False)
    return df

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    # common aliases -> lowercase for internal use
    lower_map = {c: c.lower() for c in df.columns}
    df.rename(columns=lower_map, inplace=True)
    # date
    if "date" not in df.columns:
        raise ValueError("Column 'date' not found in diagnostics CSV.")
    df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)
    df = df.dropna(subset=["date"]).sort_values("date")
    # valeur (daily intensity)
    if "valeur" not in df.columns:
        raise ValueError("Column 'valeur' not found (expected daily intensity).")
    df["valeur"] = pd.to_numeric(df["valeur"], errors="coerce").fillna(0.0)
    # thresholds (accept 'seuil_sup' from Phase 1, or older 'seuil')
    if "seuil_sup" in df.columns:
        df["seuil_k"] = pd.to_numeric(df["seuil_sup"], errors="coerce")
    elif "seuil" in df.columns:
        df["seuil_k"] = pd.to_numeric(df["seuil"], errors="coerce")
    else:
        df["seuil_k"] = np.nan
    # est_pic as boolean if present
    if "est_pic" in df.columns:
        df["est_pic"] = pd.to_numeric(df["est_pic"], errors="coerce").fillna(0).astype(int).astype(bool)
    return df

def read_anchors_csv(path: Path) -> Dict[str, pd.Timestamp]:
    """Read anchors CSV with columns: corpus, anchor_date (YYYY-MM-DD)."""
    df = read_csv_smart(path)
    # normalize
    cols = {c.lower(): c for c in df.columns}
    if "corpus" not in cols or "anchor_date" not in cols:
        raise ValueError("anchors.csv must have columns: corpus, anchor_date")
    records = {}
    for _, row in df.iterrows():
        corpus = str(row[cols["corpus"]]).strip()
        try:
            dt = pd.to_datetime(str(row[cols["anchor_date"]]), errors="raise")
        except Exception:
            raise ValueError(f"Invalid date for corpus '{corpus}' in anchors.csv: {row[cols['anchor_date']]}")
        records[corpus] = pd.Timestamp(dt.date())
    return records

# -------------------- Core logic --------------------
def list_anniversaries(anchor: pd.Timestamp, start: pd.Timestamp, end: pd.Timestamp) -> List[pd.Timestamp]:
    """All yearly recurrences of anchor between [start,end] inclusive."""
    years = range(start.year, end.year + 1)
    out = []
    for y in years:
        try:
            d = anchor.replace(year=y)
        except ValueError:
            # handle Feb 29 anchors (shift to Feb 28 if needed)
            if anchor.month == 2 and anchor.day == 29:
                d = pd.Timestamp(year=y, month=2, day=28)
            else:
                raise
        if start <= d <= end:
            out.append(d)
    return out

def window_mask(dates: pd.Series, center: pd.Timestamp, radius: int) -> pd.Series:
    """Boolean mask for dates within +/- radius days of center (inclusive)."""
    delta = (dates - center).dt.days.abs()
    return delta <= radius

def auto_anchor_from_peaks(df: pd.DataFrame) -> Optional[pd.Timestamp]:
    """Fallback: pick the date of the highest 'valeur' among flagged peaks; if not present, the global max."""
    if "est_pic" in df.columns and df["est_pic"].any():
        d = df.loc[df["est_pic"], ["date", "valeur"]].sort_values("valeur", ascending=False).head(1)
        if not d.empty:
            return pd.Timestamp(d.iloc[0]["date"].date())
    # else: global max
    d = df[["date", "valeur"]].sort_values("valeur", ascending=False).head(1)
    return pd.Timestamp(d.iloc[0]["date"].date()) if not d.empty else None

def summarize_anniversary_windows(
    df: pd.DataFrame, anniversaries: List[pd.Timestamp], window: int
) -> pd.DataFrame:
    """For each anniversary date, compute window stats and baseline."""
    rows = []
    baseline_mean = df["valeur"].mean()
    baseline_median = df["valeur"].median()
    for d in anniversaries:
        mask = window_mask(df["date"], d, window)
        win = df.loc[mask]
        days = int(mask.sum())
        sum_val = float(win["valeur"].sum())
        max_val = float(win["valeur"].max()) if not win.empty else 0.0
        days_above = int((win["valeur"] > win["seuil_k"]).sum()) if "seuil_k" in win.columns else np.nan
        rows.append({
            "year": d.year,
            "anniversary_date": d.date().isoformat(),
            "window_days": days,
            "window_radius": window,
            "sum_valeur": sum_val,
            "max_valeur": max_val,
            "days_above_threshold": days_above,
            "baseline_mean": float(baseline_mean),
            "baseline_median": float(baseline_median),
        })
    return pd.DataFrame(rows)

def slugify(text: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "-" for ch in text).strip("-")

def plot_series_with_anniversaries(
    df: pd.DataFrame,
    anniversaries: List[pd.Timestamp],
    title: str,
    outpath: Path,
    top_peaks: int = 5
) -> None:
    """Plot daily intensity, optional threshold, peak markers, and vertical anniversary lines."""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df["date"], df["valeur"], label="Daily intensity (valeur)")
    if "seuil_k" in df.columns and df["seuil_k"].notna().any():
        ax.plot(df["date"], df["seuil_k"], linestyle="--", label="Threshold (mean + K·σ)")
    # peaks
    if "est_pic" in df.columns and df["est_pic"].any():
        pics = df.loc[df["est_pic"]]
        ax.scatter(pics["date"], pics["valeur"], s=18, label="Peaks (est_pic=1)")
        # label top N peaks
        if top_peaks and top_peaks > 0:
            top = pics.sort_values("valeur", ascending=False).head(top_peaks)
            for _, r in top.iterrows():
                ax.text(r["date"], r["valeur"], r["date"].date().isoformat(),
                        fontsize=8, rotation=45, va="bottom", ha="left")
    # anniversaries
    for d in anniversaries:
        ax.axvline(d, color=None, alpha=0.4)  # default color cycle
        ax.text(d, ax.get_ylim()[1], d.date().isoformat(), rotation=90, va="top", ha="right", fontsize=8)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Intensity (valeur)")
    ax.legend(loc="best")
    fig.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=220)
    plt.close(fig)

# -------------------- CLI --------------------
def main():
    ap = argparse.ArgumentParser(description="Anniversary & reactivation analysis over viral-wave diagnostics.")
    ap.add_argument("--input", required=True, type=Path, help="Diagnostics CSV (from Phase 1).")
    ap.add_argument("--output-dir", required=True, type=Path, help="Directory for plots and summary CSVs.")
    ap.add_argument("--corpus", action="append", default=None,
                    help="Corpus to include (repeat flag for multiple). If omitted, runs for all.")
    ap.add_argument("--anchor", type=str, default=None,
                    help="Anchor date (YYYY-MM-DD) for single corpus; per-corpus use --anchors-csv.")
    ap.add_argument("--anchors-csv", type=Path, default=None,
                    help="CSV with columns: corpus,anchor_date (YYYY-MM-DD).")
    ap.add_argument("--window", type=int, default=7, help="+/- days around anniversary (default: 7).")
    ap.add_argument("--top-peaks", type=int, default=5, help="Annotate top-N peaks on the plot.")
    args = ap.parse_args()

    df = normalize_columns(read_csv_smart(args.input))

    # Choose corpora
    corpus_col = "corpus" if "corpus" in df.columns else ("Corpus".lower() if "Corpus" in df.columns else None)
    corpora = sorted(df[corpus_col].dropna().unique()) if corpus_col else ["__ALL__"]
    if args.corpus:
        wanted = set(args.corpus)
        corpora = [c for c in corpora if c in wanted]

    # Read anchors mapping if provided
    anchors_map: Dict[str, pd.Timestamp] = {}
    if args.anchors_csv:
        anchors_map = read_anchors_csv(args.anchors_csv)

    # For single corpus + --anchor
    single_anchor = pd.to_datetime(args.anchor).date() if args.anchor else None

    # Process each corpus
    summaries = []
    for corpus_name in corpora:
        if corpus_col:
            sub = df[df[corpus_col] == corpus_name].copy()
            title_prefix = corpus_name
        else:
            sub = df.copy()
            title_prefix = "All data"

        if sub.empty:
            continue

        # Decide anchor
        if args.anchors_csv and corpus_name in anchors_map:
            anchor = pd.Timestamp(anchors_map[corpus_name])
        elif single_anchor and len(corpora) == 1:
            anchor = pd.Timestamp(single_anchor)
        else:
            anchor = auto_anchor_from_peaks(sub)

        if anchor is None:
            # no anchor possible -> skip summary but still output plain series plot
            anniversaries = []
        else:
            anniversaries = list_anniversaries(anchor, sub["date"].min(), sub["date"].max())

        # Summary CSV
        if anniversaries:
            summary = summarize_anniversary_windows(sub, anniversaries, args.window)
            summary.insert(0, "corpus", corpus_name)
            summary.insert(1, "anchor", anchor.date().isoformat())
            summaries.append(summary)

        # Plot
        plot_path = args.output_dir / "plots" / f"{slugify(title_prefix)}-anniversaries.png"
        plot_title = f"{title_prefix} — peaks and anniversaries"
        plot_series_with_anniversaries(sub, anniversaries, plot_title, plot_path, top_peaks=args.top_peaks)

    # Concatenate and save all summaries
    if summaries:
        out = pd.concat(summaries, ignore_index=True)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        out.to_csv(args.output_dir / "anniversary_summary.csv", index=False, encoding="utf-8")
        print(f"Saved summary to: {args.output_dir / 'anniversary_summary.csv'}")
    else:
        print("No summaries produced (no anchors or no anniversaries in range).")
    print(f"Plots saved under: {args.output_dir / 'plots'}")


if __name__ == "__main__":
    main()
