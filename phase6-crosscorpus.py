# phase6-crosscorpus.py
# Identify pages that appear across multiple corpora (privacy-preserving).
# - relative paths (no local machine paths)
# - hashed page names (no raw identifiers in outputs)
# - outputs CSV + XLSX summaries

import os, argparse, hashlib
from pathlib import Path
import pandas as pd

# ---------- Helpers ----------

def read_resilient(path: Path) -> pd.DataFrame:
    encs = ["utf-8-sig","utf-8","cp1252","latin1","utf-16","utf-16le","utf-16be"]
    seps = [None, ",", ";", "\t", "|"]
    for enc in encs:
        for sep in seps:
            try:
                return pd.read_csv(path, sep=sep, encoding=enc, engine="python", on_bad_lines="skip")
            except Exception:
                pass
    return pd.read_excel(path)

def pick_col(df: pd.DataFrame, candidates: list[str]) -> str:
    cmap = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cmap:
            return cmap[cand.lower()]
    for c in df.columns:
        cl = c.lower()
        if any(tok.lower() in cl for tok in candidates):
            return c
    return None

def hash_page(value: str, salt: str) -> str:
    return hashlib.sha256((salt + str(value)).encode("utf-8")).hexdigest()[:12]

# ---------- CLI ----------

parser = argparse.ArgumentParser(description="Phase 6 – Cross-corpus pages (privacy-preserving).")
parser.add_argument("--tox", default=os.getenv("CSV_TOX", "data/output_toxify_allcorpus.csv"),
                    help="CSV with Post_ID + Corpus (default: data/output_toxify_allcorpus.csv)")
parser.add_argument("--fb",  default=os.getenv("CSV_FB", "data/base_fb.csv"),
                    help="CSV/Excel with Post_ID + Page info (default: data/base_fb.csv)")
parser.add_argument("--page-cols", nargs="*", default=["Page Name","Page","Account","Publisher","User"],
                    help="Candidate columns for page identifier.")
parser.add_argument("--salt", default=os.getenv("HASH_SALT", "set-your-own-salt"),
                    help="Salt for hashing page identifiers (change in production).")
parser.add_argument("--outdir", default="outputs/phase6-crosscorpus",
                    help="Output folder (default: outputs/phase6-crosscorpus).")
args = parser.parse_args()

outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

# ---------- Load ----------

tox = read_resilient(Path(args.tox))
fb  = read_resilient(Path(args.fb))

# check tox columns
required_tox = {"Post_ID","Corpus"}
if not required_tox.issubset(tox.columns):
    raise ValueError(f"Missing required columns in tox file: {required_tox - set(tox.columns)}")

# pick page col
page_col = pick_col(fb, args.page_cols)
if page_col is None:
    raise ValueError("Page column not found in fb file. Try --page-cols option.")

# ---------- Merge ----------

fb_small = fb[["Post_ID", page_col]].copy()
df = pd.merge(tox[["Post_ID","Corpus"]], fb_small, on="Post_ID", how="inner")
df = df.rename(columns={page_col: "Page"})
df = df.dropna(subset=["Page","Corpus"])
df["Page"] = df["Page"].astype(str).str.strip()
df["Corpus"] = df["Corpus"].astype(str).str.strip()

# hash page names
df["page_hash"] = df["Page"].map(lambda x: hash_page(x, args.salt))

# ---------- Count corpus coverage ----------

pages_multi = (
    df.groupby("page_hash")["Corpus"]
      .nunique()
      .reset_index(name="n_corpus_distinct")
      .sort_values(["n_corpus_distinct","page_hash"], ascending=[False,True])
)

# summary: how many pages appear in 1, 2, or 3 corpora
summary = (
    pages_multi.groupby("n_corpus_distinct")["page_hash"]
               .nunique()
               .reset_index(name="n_pages")
               .sort_values("n_corpus_distinct")
)

# list of pages present in all 3 corpora
pages_triple = pages_multi[pages_multi["n_corpus_distinct"] == 3].copy()

# example counts per corpus for triple pages
examples_triple = (
    df[df["page_hash"].isin(pages_triple["page_hash"])]
    .groupby(["page_hash","Corpus"]).size().reset_index(name="n_posts")
    .pivot(index="page_hash", columns="Corpus", values="n_posts")
    .fillna(0).astype(int)
    .sort_index()
)

# ---------- Save ----------

out_xlsx = outdir / "crosscorpus_pages.xlsx"
with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
    pages_multi.to_excel(writer, sheet_name="pages_n_corpus", index=False)
    summary.to_excel(writer, sheet_name="summary_1_2_3", index=False)
    pages_triple.to_excel(writer, sheet_name="pages_in_all3", index=False)
    examples_triple.to_excel(writer, sheet_name="examples_triple", index=True)

pages_multi.to_csv(outdir / "pages_n_corpus.csv", index=False)
summary.to_csv(outdir / "summary_1_2_3.csv", index=False)
pages_triple.to_csv(outdir / "pages_in_all3.csv", index=False)

# console quick view
print("\n=== Distribution of pages across corpora ===")
print(summary.to_string(index=False))
print(f"\nPages appearing in all 3 corpora: {len(pages_triple)}")
print(f"\nExcel saved to: {out_xlsx.resolve()}")
print(f"CSV files saved to: {outdir.resolve()}")
