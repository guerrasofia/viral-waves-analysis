#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
toxicity_analysis.py — Compute Detoxify toxicity scores over long texts using a token sliding window.

Privacy:
- By default, the output includes only toxicity columns. Use --keep-text to keep original text (not recommended for public data).
- Hugging Face token should come from the environment (HUGGINGFACE_TOKEN) or keyring, not hard-coded.

Usage example:
python scripts/toxicity_analysis.py \
  --input data/input.csv \
  --output outputs/toxicity_scores.csv \
  --text-col Message \
  --model unbiased --window 512 --stride 256
"""

from __future__ import annotations
import os
import argparse
from pathlib import Path
from typing import Dict, List
import pandas as pd
from tqdm import tqdm

try:
    from detoxify import Detoxify
    from transformers import AutoTokenizer
    from huggingface_hub import login
except ImportError as e:
    raise SystemExit(
        "Missing dependencies. Install with: pip install -r requirements.txt"
    ) from e


TOXICITY_LABELS: List[str] = [
    "toxicity", "severe_toxicity", "obscene",
    "identity_attack", "insult", "threat", "sexual_explicit"
]

MODEL_NAME_MAP = {
    "unbiased": "unitary/unbiased-toxic-roberta",
    "original": "unitary/toxic-bert",
    "multilingual": "unitary/multilingual-toxic-xlm-roberta",
}


def sliding_token_windows(text: str, tokenizer, window_size: int, stride: int):
    """Yield decoded text chunks of ~window_size tokens with stride overlap."""
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    n = len(token_ids)
    i = 0
    while i < n:
        j = min(i + window_size, n)
        chunk_ids = token_ids[i:j]
        chunk_text = tokenizer.decode(
            chunk_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        yield i, j, chunk_text
        i += stride


def analyze_text(text: str, model, tokenizer, window_size: int, stride: int) -> Dict[str, float]:
    """Return average scores per label over sliding windows."""
    if not isinstance(text, str) or not text.strip():
        return {f"toxicity_{lbl}": None for lbl in TOXICITY_LABELS}

    buckets = {lbl: [] for lbl in TOXICITY_LABELS}
    for _, _, chunk in sliding_token_windows(text, tokenizer, window_size, stride):
        scores = model.predict(chunk)
        for lbl in TOXICITY_LABELS:
            val = scores.get(lbl)
            if val is not None:
                buckets[lbl].append(float(val))

    return {
        f"toxicity_{lbl}": (sum(vals) / len(vals) if vals else None)
        for lbl, vals in buckets.items()
    }


def main():
    ap = argparse.ArgumentParser(description="Compute Detoxify toxicity scores for a CSV.")
    ap.add_argument("--input", required=True, type=Path, help="Input CSV path.")
    ap.add_argument("--output", required=True, type=Path, help="Output CSV path.")
    ap.add_argument("--text-col", default="Message", help="Column name containing text.")
    ap.add_argument("--id-col", default=None, help="Optional identifier column to keep.")
    ap.add_argument("--model", choices=list(MODEL_NAME_MAP), default="unbiased",
                    help="Detoxify variant.")
    ap.add_argument("--window", type=int, default=512, help="Token window size.")
    ap.add_argument("--stride", type=int, default=256, help="Token stride.")
    ap.add_argument("--keep-text", action="store_true",
                    help="Keep original text column in the output (not recommended for public releases).")
    args = ap.parse_args()

    # Auth (optional for public models; some environments still benefit from login)
    hf_token = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if hf_token:
        try:
            login(token=hf_token)
        except Exception:
            pass  # continue without explicit login if it fails

    pretrained = MODEL_NAME_MAP[args.model]
    tokenizer = AutoTokenizer.from_pretrained(pretrained, use_fast=True)
    model = Detoxify(args.model)

    # Read CSV with dialect sniffing
    df = pd.read_csv(args.input, sep=None, engine="python", encoding="utf-8", on_bad_lines="skip")

    if args.text_col not in df.columns:
        raise SystemExit(f"Column '{args.text_col}' not found in {args.input}.")

    # Prepare result frame with minimal columns
    keep_cols = [c for c in [args.id_col] if c and c in df.columns]
    if args.keep_text and args.text_col not in keep_cols:
        keep_cols.append(args.text_col)

    out = df[keep_cols].copy() if keep_cols else pd.DataFrame(index=df.index)

    # Compute toxicity
    tqdm.pandas(desc="Scoring toxicity")
    scores = df[args.text_col].progress_apply(
        lambda txt: analyze_text(str(txt) if pd.notna(txt) else "", model, tokenizer, args.window, args.stride)
    )
    scores_df = pd.DataFrame(list(scores.values), index=df.index)

    result = pd.concat([out, scores_df], axis=1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False, encoding="utf-8")
    print(f"Saved toxicity scores to: {args.output}")


if __name__ == "__main__":
    main()
