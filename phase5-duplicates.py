# phase5-duplicates.py
# Detect repeated/duplicated messages per corpus (privacy-preserving).
# - relative paths (no local machine paths)
# - hashed message text (no raw text in outputs by default)
# - optional truncated sample for qualitative inspection

import os, re, argparse, hashlib
from pathlib import Path
import pandas as pd
import numpy as np

# ---------- Helpers ----------

def read_resilient(path: Path) -> pd.DataFrame:
    encs = ["utf-8-sig","utf-8","cp1252","latin1","utf-16","utf-16le","utf-16be"]
    seps = [None, ",", ";", "\t", "|"]
    for enc in encs:
        for sep in seps:
            try:
                return pd.read_csv(path, encoding=enc, sep=sep, engine="python", on_bad_lines="skip")
            except Exception:
                pass
    # fallback to Excel
    return pd.read_excel(path)

def pick_first_existing(df: pd.DataFrame, candidates: list[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"None of the expected columns found: {candidates}")

def normalize_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.strip()
    s = re.sub(r"\s+", " ", s)                          # collapse whitespace
    s = re.sub(r"https?://\S+", "<url>", s, flags=re.I) # mask URLs
    return s

def text_hash(s: str, salt: str) -> str:
    return hashlib.sha256((salt + s).encode("utf-8")).hexdigest()[:16]

def truncate(s: str, n: int = 140) -> str:
    if not isinstance(s, str):
        return ""
    return s if len(s) <= n else s[: n - 1] + "…"

# ---------- CLI ----------

parser = argparse.ArgumentParser(
    description="Detect repeated messages per corpus (privacy-preserving)."
)
parser.add_argument("--tox", default=os.getenv("CSV_TOX", "data/output_toxify_allcorpus.csv"),
                    help="CSV with Post_ID and Corpus (default: data/output_toxify_allcorpus.csv)")
parser.add_argument("--fb",  default=os.getenv("CSV_FB", "data/base_fb.csv"),
                    help="CSV/Excel with Post_ID and a text column (default: data/base_fb.csv)")
parser.add_argument("--text-cols", nargs="*", default=["Message","Text","Description","message","text","description"],
                    help="Candidate text columns to search in the FB file.")
parser.add_argument("--min-repeats", type=int, default=2,
                    help="Minimum number of repeats to keep (default: 2).")
parser.add_argument("--salt", default=os.getenv("HASH_SALT", "set-your-own-salt"),
                    help="Salt used for hashing text (change in production).")
parser.add_argument("--include-sample", action="store_true",
                    help="If set, include a truncated sample of one message (privacy trade-off).")
parser.add_argument("--sample-len", type=int, default=140,
                    help="Max length of truncated sample if included (default: 140).")
parser.add_argument("--outdir", default="outputs",
                    help="Output folder (default: outputs).")
args = parser.parse_args()

outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

# ---------- Load ----------

tox = read_resilient(Path(args.tox))
fb  = read_resilient(Path(args.fb))

if "Post_ID" not in tox.columns:
    raise ValueError("`Post_ID` not found in tox file.")
if "Corpus" not in tox.columns:
    alt = pick_first_existing(tox, ["corpus","CORPUS"])
    tox = tox.rename(columns={alt: "Corpus"})

text_col = pick_first_existing(fb, args.text_cols)
if "Post_ID" not in fb.columns:
    raise ValueError("`Post_ID` not found in fb file.")

# ---------- Merge & normalize ----------

df = pd.merge(tox[["Post_ID","Corpus"]], fb[["Post_ID", text_col]], on="Post_ID", how="inner")
df = df.rename(columns={text_col: "Text"})
df["Text_norm"] = df["Text"].fillna("").map(normalize_text)

# hash normalized text (no raw text in outputs)
df["text_hash"] = df["Text_norm"].map(lambda s: text_hash(s, args.salt))

# ---------- Count duplicates per corpus ----------

dupes = (
    df.groupby(["Corpus","text_hash"], as_index=False)
      .agg(
          n_posts=("Post_ID", "count"),
          sample=lambda g: truncate(g["Text"].dropna().astype(str).iloc[0], args.sample_len)
      )
)

if not args.include_sample:
    dupes = dupes.drop(columns=["sample"], errors="ignore")

dupes = dupes[dupes["n_posts"] >= args.min_repeats].sort_values(
    ["Corpus","n_posts"], ascending=[True, False]
)

# ---------- Save ----------

out_csv = outdir / "repeated_messages_by_corpus.csv"
dupes.to_csv(out_csv, index=False)

print("[OK] Repeated messages table saved to:", out_csv)
print(dupes.head(15))
