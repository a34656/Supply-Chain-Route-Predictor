import os
import pandas as pd
import chardet
import re

TARGET_COLUMNS = {
    "economy" : "country",
    "country" : "country",
    "lpi score" : "lpi_score",
    "lpi grouped rank" : "lpi_rank",
    "customs score" : "customs_score",
    "customs grouped rank" : "customs_rank",
    "infrastructure score" : "infrastructure_score",
    "infrastructure grouped rank" : "infrastructure_rank",
    "international shipments score": "shipments_score",
    "international shipments grouped rank": "shipments_rank",
    "logistics competence and quality score": "logistics_score",
    "logistics competence and quality grouped rank": "logistics_rank",
    "timeliness score": "timeliness_score",
    "timeliness grouped rank": "timeliness_rank",
    "tracking and tracing score": "tracking_score",
    "tracking and tracing grouped rank": "tracking_rank"
}

def normalise_column(colname : str) -> str:
    colname = colname.strip().lower()
    colname = re.sub(r"[\s/]+", " ", colname)
    return colname

def detect_encoding(path):
    with open(path, "rb") as f:
        raw = f.read(5000)
    return chardet.detect(raw)['encoding']

def clean_single_lpi(path, year):
    print(f"\nClesaning LPI for year {year} from {path}")

    if path.endswith(".xlsx"):
        df = pd.read_excel(path)
    else:
        enc = detect_encoding(path)
        df = pd.read_csv(path, encoding=enc)
    
    df.columns = [normalise_column(c) for c in df.columns]

    mapped_cols = {}
    for c in df.columns:
        if c in TARGET_COLUMNS:
            mapped_cols[c] = TARGET_COLUMNS[c]
    
    df = df[list(mapped_cols.keys())].rename(columns = mapped_cols)

    for tgt in TARGET_COLUMNS.values():
        if tgt not in df.columns:
            df[tgt] = None
    
    score_cols = [c for c in df.columns if "score" in c]
    rank_cols= [c for  c in df. columns if "rank" in c]

    for col in score_cols + rank_cols:
        df[col] = (df[col].astype(str).str.replace(",", "").str.replace("…", "").str.strip().replace("", None))
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["country"] = (df["country"].astype(str).str.replace(r"\*|\d+", "", regex=True).str.strip())
    df["year"] = year
    return df
