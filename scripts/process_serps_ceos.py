#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Process daily CEO SERPs with sentiment & control classification,
and write both row-level and aggregate outputs for the dashboard.

Inputs
------
Raw S3 CSV per day:
  https://tk-public-data.s3.us-east-1.amazonaws.com/serp_files/{date}-ceo-serps.csv
  NOTE: In raw files, the column named "company" holds the alias text
        (e.g., "Tim Cook Apple"), not the canonical company.

Local maps:
  rosters/main-roster.csv (CEO, Company, CEO Alias, Websites) -> primary source

Outputs
-------
Row-level processed SERPs (modal):
  data/processed_serps/{date}-ceo-serps-modal.csv

Per-CEO daily aggregate:
  data/processed_serps/{date}-ceo-serps-table.csv

Rolling index (dashboard table & SERP trend):
  data/daily_counts/ceo-serps-daily-counts-chart.csv

Usage
-----
python scripts/process_serps_ceos.py --date 2025-09-17
python scripts/process_serps_ceos.py --backfill 2025-09-15 2025-09-30
(no args) -> tries today, then yesterday (skips if before FIRST_AVAILABLE_DATE)
"""

from __future__ import annotations
import argparse
import io
import re
import sys
import datetime as dt
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# --------------------------- Config ---------------------------

S3_TEMPLATE = "https://tk-public-data.s3.us-east-1.amazonaws.com/serp_files/{date}-ceo-serps.csv"

# First day CEO SERPs exist
FIRST_AVAILABLE_DATE = dt.date(2025, 9, 15)

# Updated to use consolidated roster
MAIN_ROSTER_PATH = Path("rosters/main-roster.csv")

# Updated paths - all CEO SERP files consolidated in data/processed_serps
OUT_DIR_ROWS = Path("data/processed_serps")
OUT_DIR_DAILY = Path("data/processed_serps")
INDEX_DIR = Path("data/daily_counts")
INDEX_PATH = INDEX_DIR / "ceo-serps-daily-counts-chart.csv"

for p in (OUT_DIR_ROWS, OUT_DIR_DAILY, INDEX_DIR):
    p.mkdir(parents=True, exist_ok=True)

# Control rules
CONTROLLED_SOCIAL_DOMAINS = {
    "facebook.com", "linkedin.com", "instagram.com", "twitter.com", "x.com"
}
CONTROLLED_PATH_KEYWORDS = {
    "/leadership/", "/about/", "/governance/", "/team/", "/investors/", "/board-of-directors", "/members/", "/member/" 
}
UNCONTROLLED_DOMAINS = {
    "wikipedia.org", "youtube.com", "youtu.be", "tiktok.com"
}

# Words/phrases to ignore for title-based sentiment classification
NEUTRALIZE_TITLE_TERMS = [
    r"\bflees\b",
    r"\bsavage\b",
    r"\brob\b",
    r"\bnicholas\s+lower\b",
    r"\bmad\s+money\b",
]
NEUTRALIZE_TITLE_RE = re.compile("|".join(NEUTRALIZE_TITLE_TERMS), flags=re.IGNORECASE)

ALWAYS_NEGATIVE_TERMS = [
    r"\bpaid\b", r"\bcompensation\b", r"\bpay\b",
    r"\bmandate\b",
    r"\bexit(s)?\b", r"\bstep\s+down\b", r"\bsteps\s+down\b", r"\bremoved\b",
    r"\bstill\b",
    r"\bturnaround\b",
    r"\bface\b", r"\baccused\b", r"\bcommitted\b",
    r"\baware\b",
    r"\bloss\b", r"\bdivorce\b", r"\bbankruptcy\b",
    r"\bunion\s+buster\b",
    r"\bfired\b",
    r"\bfiring\b",
    r"(?<!t)\bax(e|ed|es)?\b",
    r"\bsack(ed|s)?\b",
    r"\boust(ed)?\b",
]

ALWAYS_NEGATIVE_RE = re.compile("|".join(ALWAYS_NEGATIVE_TERMS), re.IGNORECASE)

def _should_force_negative_title(title: str) -> bool:
    return bool(ALWAYS_NEGATIVE_RE.search(title or ""))

# ----------------------- Force Negative Titles Helper ---------------
def _should_neutralize_title(title: str) -> bool:
    """Return True if the title contains terms that should neutralize sentiment."""
    return bool(NEUTRALIZE_TITLE_RE.search(str(title or "")))

# ------------------------ Small helpers -----------------------

def strip_neutral_terms_from_title(title: str) -> str:
    s = str(title or "")
    s = NEUTRALIZE_TITLE_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def norm(s: str) -> str:
    s = str(s or "").lower().strip()
    s = re.sub(r"[^a-z0-9\s]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

LEGAL_SUFFIXES = {"inc", "inc.", "corp", "co", "co.", "llc", "plc", "ltd", "ltd.", "ag", "sa", "nv"}

def simplify_company(s: str) -> str:
    toks = norm(s).split()
    toks = [t for t in toks if t not in LEGAL_SUFFIXES]
    return " ".join(toks)

def _norm_token(s: str) -> str:
    """Normalize a token to alphanumeric characters only (lowercase).
    
    Used for matching company/brand names within domain parts.
    Examples:
      "Apple" -> "apple"
      "Apple Inc." -> "appleinc"
      "123-Brand" -> "123brand"
    """
    return "".join(ch for ch in (s or "").lower() if ch.isalnum())

def read_csv_safely(text_or_path):
    try:
        if isinstance(text_or_path, str) and "\n" in text_or_path:
            return pd.read_csv(io.StringIO(text_or_path))
        return pd.read_csv(text_or_path, encoding="utf-8-sig")
    except Exception:
        if isinstance(text_or_path, str) and "\n" in text_or_path:
            return pd.read_csv(io.StringIO(text_or_path), engine="python")
        return pd.read_csv(text_or_path, engine="python", encoding="utf-8-sig")

def fetch_csv_text(url: str, timeout=30):
    r = requests.get(url, timeout=timeout)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    r.encoding = r.apparent_encoding or "utf-8"
    return r.text

def load_roster_data():
    """
    Load roster data with support for pipe-separated URLs.
    Example: 'apple.com|support.apple.com|developer.apple.com'
    """
    if not MAIN_ROSTER_PATH.exists():
        raise FileNotFoundError(f"Main roster not found: {MAIN_ROSTER_PATH}")

    df = read_csv_safely(MAIN_ROSTER_PATH)
    cols = {c.strip().lower(): c for c in df.columns}
    
    def col(*names):
        for name in names:
            for k, v in cols.items():
                if k == name.lower():
                    return v
        return None

    ceo_col = col("ceo")
    company_col = col("company")
    alias_col = col("ceo alias", "alias")
    website_col = col("website", "Websites", "domain", "url")

    if not (ceo_col and company_col):
        raise ValueError("Main roster must have CEO and Company columns")

    ceo_to_company = {}
    for _, row in df.iterrows():
        ceo = str(row[ceo_col]).strip()
        company = str(row[company_col]).strip()
        if ceo and company and ceo != "nan" and company != "nan":
            ceo_to_company[ceo] = company

    alias_map = {}
    if alias_col:
        for _, row in df.iterrows():
            alias = str(row[alias_col]).strip()
            ceo = str(row[ceo_col]).strip()
            company = str(row[company_col]).strip()
            if alias and ceo and company and alias != "nan":
                alias_map[norm(alias)] = (ceo, company)

    for ceo, comp in ceo_to_company.items():
        alias_map.setdefault(norm(f"{ceo} {comp}"), (ceo, comp))

    controlled_domains = {}  # Changed to dict: company -> Set[domains]
    if website_col:
        print(f"[INFO] Loading controlled domains from roster ({website_col} column)...")
        for idx, row in df.iterrows():
            company = str(row[company_col]).strip() if company_col else ""
            if not company or company.lower() == "nan":
                continue
            
            val = row[website_col]
            # Handle NaN values properly
            if pd.isna(val):
                continue
            
            val = str(val).strip()
            if not val or val.lower() == "nan":
                continue
            
            # Initialize set for this company if not exists
            if company not in controlled_domains:
                controlled_domains[company] = set()
            
            # Split on pipe character to support multiple URLs per company
            # Works for both "apple.com" (single) and "apple.com|support.apple.com" (multiple)
            urls = val.split("|")
            
            for url in urls:
                url = url.strip()
                if not url or url.lower() == "nan":
                    continue
                try:
                    if not url.startswith(("http://", "https://")):
                        url = f"https://{url}"
                    parsed = urlparse(url)
                    host = (parsed.netloc or parsed.path or "").lower().strip()
                    host = host.replace("www.", "")
                    if host and "." in host:
                        controlled_domains[company].add(host)
                        print(f"  ✓ {company}: {host}")
                except Exception as e:
                    print(f"  ⚠️  Failed to parse {url}: {e}")
        
        total_domains = sum(len(domains) for domains in controlled_domains.values())
        print(f"[OK] Loaded {len(controlled_domains)} companies with {total_domains} total controlled domains")

    return alias_map, ceo_to_company, controlled_domains

# -------------------- Normalization & rules --------------------

def normalize_raw_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower(): c for c in df.columns}
    q_c   = cols.get("company") or cols.get("query") or cols.get("search")
    t_c   = cols.get("title") or cols.get("page_title") or cols.get("result")
    u_c   = cols.get("url") or cols.get("link")
    p_c   = cols.get("position") or cols.get("rank") or cols.get("pos")
    sn_c  = cols.get("snippet") or cols.get("description")

    out = pd.DataFrame()
    out["query_alias"] = df[q_c].astype(str).str.strip() if q_c else ""
    out["title"]       = df[t_c].astype(str).str.strip() if t_c else ""
    out["url"]         = df[u_c].astype(str).str.strip() if u_c else ""
    out["position"]    = pd.to_numeric(df[p_c], errors="coerce") if p_c else pd.Series([None]*len(df))
    out["snippet"]     = df[sn_c].astype(str).str.strip() if sn_c else ""
    return out

def resolve_ceo_company(query_alias: str, alias_map, ceo_to_company):
    qn = norm(query_alias)
    if qn in alias_map:
        return alias_map[qn]

    best = None
    best_score = 0
    for ceo, comp in ceo_to_company.items():
        tokens = set(f"{norm(ceo)} {simplify_company(comp)}".split())
        if tokens.issubset(set(qn.split())):
            score = len(tokens)
            if score > best_score:
                best = (ceo, comp)
                best_score = score
    return best if best else ("", "")

def classify_control(url: str, position, company: str, company_domains):
    """
    Classify if a URL is controlled by a company using multiple rules:
    (0) Rule 0: Explicitly uncontrolled domains (return False immediately)
    (1) Rule 1: Always-controlled social platforms
    (2) Rule 2: Company-specific domains from roster (ONLY this company's domains)
    (3) Rule 3: Domain contains the company token
    (4) Rule 4: Controlled path keywords
    """
    try:
        parsed = urlparse(url or "")
        domain = (parsed.netloc or "").lower().replace("www.", "")
        path   = (parsed.path or "").lower()
    except Exception:
        domain, path = "", ""

    # Rule 0: Explicitly uncontrolled domains (skip everything else)
    if any(d == domain or domain.endswith("." + d) for d in UNCONTROLLED_DOMAINS):
        return False

    # Rule 1: Always-controlled social platforms (and their subdomains)
    for social_domain in CONTROLLED_SOCIAL_DOMAINS:
        if domain == social_domain or domain.endswith("." + social_domain):
            return True

    # Rule 2: Company-specific domains from roster (ONLY this company's domains)
    company_specific_domains = company_domains.get(company, set())
    for roster_domain in company_specific_domains:
        if domain == roster_domain or domain.endswith("." + roster_domain):
            return True

    # Rule 3: Domain contains the company token (proper subdomain matching)
    comp_simple = simplify_company(company)
    if comp_simple:
        # Split hostname into parts and normalize each part
        # For "news.apple.com": parts are ["news", "apple", "com"]
        domain_parts = domain.split('.')
        normalized_parts = [_norm_token(part) for part in domain_parts if part]
        comp_token = _norm_token(comp_simple)
        
        # Check if company token matches any part of the domain (except TLD)
        # This prevents false positives like substring matching
        if comp_token in normalized_parts[:-1]:  # Exclude the TLD (last part)
            return True

    # Rule 4: Controlled path keywords (e.g., /leadership/, /about/)
    if any(k in path for k in CONTROLLED_PATH_KEYWORDS):
        return True

    return False

def vader_label(analyzer, row):
    raw_text = (row.get("title") or "").strip()
    text = strip_neutral_terms_from_title(raw_text)
    if not text:
        return "neutral"
    score = analyzer.polarity_scores(text)["compound"]
    if score >= 0.05:
        return "positive"
    if score <= -0.15:
        return "negative"
    return "neutral"

# ---------------------------- Core ----------------------------

def process_one_date(date_str: str, alias_map, ceo_to_company, company_domains):
    day = dt.date.fromisoformat(date_str)
    if day < FIRST_AVAILABLE_DATE:
        print(f"[skip] {date_str} < first available ({FIRST_AVAILABLE_DATE})")
        return None

    url = S3_TEMPLATE.format(date=date_str)
    print(f"[fetch] {url}")
    text = fetch_csv_text(url)
    if text is None:
        print(f"[missing] No S3 file for {date_str}")
        return None

    raw = read_csv_safely(text)
    base = normalize_raw_columns(raw)

    mapped = base.copy()
    mapped[["ceo", "company"]] = mapped.apply(
        lambda r: pd.Series(resolve_ceo_company(r["query_alias"], alias_map, ceo_to_company)),
        axis=1,
    )

    analyzer = SentimentIntensityAnalyzer()

    from urllib.parse import urlparse  # already imported at the top

    def _sentiment_with_rules(r):
        t = str(r.get("title", "") or "")
        u = str(r.get("url", "") or "").strip().lower()
    
        # --- NEW RULE: force reddit.com to negative ---
        try:
            domain = urlparse(u).netloc.lower().replace("www.", "")
            if domain == "reddit.com" or domain.endswith(".reddit.com"):
                return "negative"
        except Exception:
            pass
    
        # Existing rules
        if _should_force_negative_title(t):
            return "negative"
        if _should_neutralize_title(t):
            return "neutral"
        return vader_label(analyzer, r)

    mapped["sentiment"] = mapped.apply(_sentiment_with_rules, axis=1)

    mapped["controlled"] = mapped.apply(lambda r: classify_control(r["url"], r["position"], r["company"], company_domains), axis=1)

    mapped.loc[mapped["controlled"] == True, "sentiment"] = "positive"

    rows_df = pd.DataFrame({
        "date":      date_str,
        "ceo":       mapped["ceo"],
        "company":   mapped["company"],
        "title":     mapped["title"],
        "url":       mapped["url"],
        "position":  mapped["position"],
        "snippet":   mapped["snippet"],
        "sentiment": mapped["sentiment"],
        "controlled":mapped["controlled"],
    })
    rows_path = OUT_DIR_ROWS / f"{date_str}-ceo-serps-modal.csv"
    rows_df.to_csv(rows_path, index=False)
    print(f"[write] {rows_path}")

    def majority_company(series):
        s = pd.Series(series).replace("", pd.NA).dropna()
        if s.empty:
            return ""
        return s.mode().iloc[0]

    ag = mapped.groupby("ceo", dropna=False).agg(
        total=("sentiment", "size"),
        controlled=("controlled", "sum"),
        negative_serp=("sentiment", lambda s: (s == "negative").sum()),
        neutral_serp=("sentiment",  lambda s: (s == "neutral").sum()),
        positive_serp=("sentiment", lambda s: (s == "positive").sum()),
        company=("company", majority_company),
    ).reset_index()
    ag.insert(0, "date", date_str)

    day_path = OUT_DIR_DAILY / f"{date_str}-ceo-serps-table.csv"
    ag.to_csv(day_path, index=False)
    print(f"[write] {day_path}")

    if INDEX_PATH.exists():
        idx = read_csv_safely(INDEX_PATH)
        idx = idx[idx["date"] != date_str]
        idx = pd.concat([idx, ag], ignore_index=True)
    else:
        idx = ag

    idx["date"] = pd.to_datetime(idx["date"], errors="coerce")
    idx = idx.sort_values(["date", "ceo"]).reset_index(drop=True)
    idx["date"] = idx["date"].dt.strftime("%Y-%m-%d")
    idx.to_csv(INDEX_PATH, index=False)
    print(f"[update] {INDEX_PATH} ({len(idx)} rows total)")

    return day_path

def backfill(start: str, end: str, alias_map, ceo_to_company, company_domains):
    d0 = dt.date.fromisoformat(start)
    d1 = dt.date.fromisoformat(end)
    if d0 > d1:
        d0, d1 = d1, d0
    d = d0
    while d <= d1:
        process_one_date(d.isoformat(), alias_map, ceo_to_company, company_domains)
        d += dt.timedelta(days=1)

def main():
    ap = argparse.ArgumentParser(description="Process CEO SERPs with sentiment/control.")
    ap.add_argument("--date", help="Process a single date (YYYY-MM-DD).")
    ap.add_argument("--backfill", nargs=2, metavar=("START", "END"),
                    help="Process an inclusive date range (YYYY-MM-DD YYYY-MM-DD).")
    args = ap.parse_args()

    alias_map, ceo_to_company, company_domains = load_roster_data()

    if args.date:
        process_one_date(args.date, alias_map, ceo_to_company, company_domains)
    elif args.backfill:
        backfill(args.backfill[0], args.backfill[1], alias_map, ceo_to_company, company_domains)
    else:
        today = dt.date.today()
        for cand in (today, today - dt.timedelta(days=1)):
            if process_one_date(cand.isoformat(), alias_map, ceo_to_company, company_domains):
                break

if __name__ == "__main__":
    sys.exit(main() or 0)
