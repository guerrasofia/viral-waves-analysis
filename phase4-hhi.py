# hhi_pages_privacy.py
# Analyze monthly concentration (HHI) of pages and aggregated rankings,
# preserving privacy (hashed IDs, no local paths).

import os, argparse, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ------------------
# Helpers
# ------------------
def read_csv_resilient(p: Path) -> pd.DataFrame:
    """Try multiple encodings and separators until CSV/Excel loads successfully."""
    encodings = ["utf-8-sig","utf-8","cp1252","latin1","utf-16","utf-16le","utf-16be"]
    separators = [None, ",", ";", "\t", "|"]
    for enc in encodings:
        for sep in separators:
            try:
                return pd.read_csv(p, sep=sep, engine="python", encoding=enc, on_bad_lines="skip")
            except Exception:
                pass
    # fallback: Excel
    return pd.read_excel(p)

def hash_id(value: str, salt: str) -> str:
    """Hash page identifiers to preserve privacy."""
    s = (salt or "default_salt").encode("utf-8")
    return hashlib.sha256(s + str(value).encode("utf-8")).hexdigest()[:12]

def p90(x):
    x = pd.to_numeric(x, errors="coerce").dropna()
    return float(np.percentile(x, 90)) if len(x) else 0.0

# ------------------
# CLI
# ------------------
parser = argparse.ArgumentParser(description="Monthly HHI analysis with privacy protection.")
parser.add_argument("--data", type=str, default=os.getenv("DATA_FILE", "data/base_fb.csv"),
                    help="Path to CSV/Excel file (default: data/base_fb.csv).")
parser.add_argument("--corpus", type=str, default=os.getenv("FILTER_CORPUS"),
                    help="Filter by corpus (optional).")
parser.add_argument("--platform", type=str, default=os.getenv("FILTER_PLATFORM"),
                    help="Filter by platform (Facebook/Instagram) (optional).")
parser.add_argument("--start", type=str, default=os.getenv("START_DATE"),
                    help="Start date YYYY-MM-DD (optional).")
parser.add_argument("--end", type=str, default=os.getenv("END_DATE"),
                    help="End date YYYY-MM-DD (optional).")
parser.add_argument("--topn", type=int, default=int(os.getenv("TOP_N", "20")),
                    help="Top N pages for ranking output (default: 20).")
parser.add_argument("--salt", type=str, default=os.getenv("HASH_SALT", "set-your-own-salt"),
                    help="Salt for page hashing (change in production).")
args = parser.parse_args()

# ------------------
# Load & prepare
# ------------------
in_path = Path(args.data)
out_dir = Path("outputs"); out_dir.mkdir(parents=True, exist_ok=True)

df = read_csv_resilient(in_path)
df.columns = [c.strip() for c in df.columns]

# date
date_candidates = ["Post Created", "Post Created Date", "post_created", "date"]
date_col = next((c for c in date_candidates if c in df.columns), None)
if not date_col:
    raise ValueError("Date column not found (try 'Post Created').")
df["date"] = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)
df = df.dropna(subset=["date"])

# interactions
inter_candidates = ["Total Interactions", "Total interactions", "Total Interactions2", "total_interactions"]
col_inter = next((c for c in inter_candidates if c in df.columns), None)
if not col_inter:
    raise ValueError("Interaction column not found.")
df["total_inter"] = pd.to_numeric(df[col_inter], errors="coerce").fillna(0)

# filters
if args.corpus and "Corpus" in df.columns:
    df = df[df["Corpus"] == args.corpus]
if args.platform and ("Plateforme" in df.columns or "platform" in df.columns):
    plat_col = "Plateforme" if "Plateforme" in df.columns else "platform"
    df = df[df[plat_col] == args.platform]
if args.start:
    df = df[df["date"] >= pd.to_datetime(args.start)]
if args.end:
    df = df[df["date"] <= pd.to_datetime(args.end)]
if df.empty:
    raise ValueError("No data left after filters.")

# page key (prefer IDs)
key_candidates = ["Facebook Id", "facebook_id", "Page ID", "page_id", "Account", "page_name", "Page Name"]
key_col = next((c for c in key_candidates if c in df.columns), None)
if not key_col:
    key_col = "page_fallback"
    df[key_col] = "page_" + df.index.astype(str)

# hash and drop sensitive columns
df["page_hash"] = df[key_col].apply(lambda x: hash_id(x, args.salt))
cols_to_drop = [c for c in ["Facebook Id","facebook_id","Page Name","page_name","Account"] if c in df.columns]
df = df.drop(columns=cols_to_drop, errors="ignore")

# ------------------
# Monthly HHI (aggregated only)
# ------------------
df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()
monthly_page = (df.groupby(["month","page_hash"], dropna=False)["total_inter"]
                  .sum().reset_index(name="inter_month"))
tot_month = monthly_page.groupby("month")["inter_month"].sum().rename("tot_month")
monthly_page = monthly_page.merge(tot_month, on="month", how="left")
monthly_page["share"] = np.where(monthly_page["tot_month"]>0,
                                 monthly_page["inter_month"]/monthly_page["tot_month"], 0.0)
hhi_monthly = (monthly_page.groupby("month")["share"].apply(lambda s: float((s**2).sum()))
               .reset_index(name="HHI").sort_values("month"))

# save HHI
hhi_path = out_dir / "hhi_monthly.csv"
hhi_monthly.to_csv(hhi_path, index=False)

# ------------------
# Aggregated page ranking (hashed IDs only)
# ------------------
agg = (df.groupby("page_hash", dropna=False)
         .agg(
             nb_posts=("date","size"),
             interactions_total=("total_inter","sum"),
             interactions_mean=("total_inter","mean"),
             interactions_median=("total_inter","median"),
             interactions_p90=("total_inter", p90),
             first_month=("month","min"),
             last_month=("month","max"),
           )
         .reset_index())

# global share of voice
total_all = float(agg["interactions_total"].sum())
agg["share_pct"] = (agg["interactions_total"] / total_all * 100).round(2) if total_all else 0.0

rank_path = out_dir / "page_ranking_aggregated.csv"
agg.sort_values("interactions_total", ascending=False).head(args.topn).to_csv(rank_path, index=False)

# ------------------
# Simple HHI plot (no sensitive info)
# ------------------
plt.figure(figsize=(12,4.5))
plt.plot(hhi_monthly["month"], hhi_monthly["HHI"], linewidth=2)
title_bits = []
if args.corpus: title_bits.append(args.corpus)
if args.platform: title_bits.append(args.platform)
plt.title("Monthly HHI" + (" - " + " | ".join(title_bits) if title_bits else ""))
plt.xlabel("Month"); plt.ylabel("HHI"); plt.grid(True, alpha=0.25)
ax = plt.gca()
ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=6, maxticks=12))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
plt.tight_layout()
plt.savefig(out_dir / "hhi_monthly.png", dpi=180)
plt.close()

print(f"[OK] Monthly HHI saved to: {hhi_path}")
print(f"[OK] Aggregated ranking (hashed) saved to: {rank_path}")
