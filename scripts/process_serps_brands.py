#!/usr/bin/env python3
"""
Process daily BRAND SERP data:

- Fetch raw SERPs from S3: https://tk-public-data.s3.us-east-1.amazonaws.com/serp_files/{date}-brand-serps.csv
- Classify sentiment (VADER) using ONLY the page TITLE
- Classify CONTROL using three rules:
    (1) Always-controlled platforms (and their subdomains):
        facebook.com, instagram.com, twitter.com, x.com, linkedin.com, play.google.com, apps.apple.com
    (2) Any domain (or its subdomains) present in rosters/main-roster.csv (Websites column)
        Multiple URLs can be separated by pipe (|) character
    (3) Domain contains the normalized brand token

- If CONTROLLED and FORCE_POSITIVE_IF_CONTROLLED = True -> sentiment is forced to "positive"

Outputs:
  1) Row-level processed SERPs:       data/processed_serps/{date}-brand-serps-modal.csv
  2) Per-company daily aggregate:     data/processed_serps/{date}-brand-serps-table.csv
  3) Rolling daily index:             data/daily_counts/brand-serps-daily-counts-chart.csv
"""

from __future__ import annotations

import argparse
import csv
import io
import re
import os
from datetime import datetime
from typing import Dict, Tuple, Set
from urllib.parse import urlparse

import pandas as pd
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# -----------------------
# Config / constants
# -----------------------
S3_URL_TEMPLATE = (
    "https://tk-public-data.s3.us-east-1.amazonaws.com/serp_files/{date}-brand-serps.csv"
)

# Updated to use consolidated roster
MAIN_ROSTER_PATH = "rosters/main-roster.csv"

# Updated paths - consolidated in data/processed_serps
OUT_ROWS_DIR = "data/processed_serps"
OUT_DAILY_DIR = "data/processed_serps"
OUT_ROLLUP = "data/daily_counts/brand-serps-daily-counts-chart.csv"

FORCE_POSITIVE_IF_CONTROLLED = True

ALWAYS_CONTROLLED_DOMAINS: Set[str] = {
    "facebook.com",
    "instagram.com",
    "twitter.com",
    "x.com",
    "linkedin.com",
    "play.google.com",
    "apps.apple.com",
}

# Words/phrases to ignore for title-based sentiment classification
# (Customize this list for BRANDS—examples shown)
NEUTRALIZE_TITLE_TERMS = [
    r"\bkilled\b",
    r"\bmlm\b",
    r"\bmad\s+money\b",
    r"\brate\s+cut\b",
    r"\bone\s+stop\s+shop\b",
    r"\bfuneral\b",
    r"\bcremation\b",
    r"\bcemetery\b",
]
NEUTRALIZE_TITLE_RE = re.compile("|".join(NEUTRALIZE_TITLE_TERMS), flags=re.IGNORECASE)

# Force-negative if the title mentions legal trouble
LEGAL_TROUBLE_TERMS = [
    r"\blawsuit(s)?\b", r"\bsued\b",
    r"\bsettlement(s)?\b", r"\bfine(d)?\b", r"\bclass[- ]action\b",
    r"\bftc\b", r"\bsec\b", r"\bdoj\b", r"\bcfpb\b"
    r"\bantitrust\b", r"\bban(s|ed)?\b"
    r"\brecall\b",
    r"\blayoffs\b",r"\bexit(s)?\b", r"\bstep\s+down\b", r"\bsteps\s+down\b",
    r"\bprobe(s|d)?\b", r"\binvestigation(s)?\b",
    r"\bsanction(s|ed)?\b", r"\bpenalt(y|ies)\b",
    r"\bfraud\b", r"\bembezzl(e|ement)\b", r"\baccused\b", r"\bcommitted\b"
    r"\bdivorce\b", r"\bbankcruptcy\b",
]
LEGAL_TROUBLE_RE = re.compile("|".join(LEGAL_TROUBLE_TERMS), flags=re.IGNORECASE)

def _title_mentions_legal_trouble(title: str) -> bool:
    return bool(LEGAL_TROUBLE_RE.search(title or ""))

def _should_neutralize_title(title: str) -> bool:
    return bool(NEUTRALIZE_TITLE_RE.search(title or ""))
    
# -----------------------
# Argument parsing / dates
# -----------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Process daily brand SERPs.")
    ap.add_argument("--date", help="YYYY-MM-DD (defaults to today)", default=None)
    return ap.parse_args()

def get_target_date(arg_date: str | None) -> str:
    if arg_date:
        try:
            datetime.strptime(arg_date, "%Y-%m-%d")
            return arg_date
        except ValueError:
            pass
    return datetime.utcnow().strftime("%Y-%m-%d")

# -----------------------
# Files / I/O helpers
# -----------------------
def ensure_dirs() -> None:
    os.makedirs(OUT_ROWS_DIR, exist_ok=True)
    os.makedirs(OUT_DAILY_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(OUT_ROLLUP), exist_ok=True)

def fetch_csv_from_s3(url: str) -> pd.DataFrame | None:
    try:
        resp = requests.get(url, timeout=45)
        resp.raise_for_status()
        return pd.read_csv(io.StringIO(resp.text))
    except Exception as e:
        print(f"[WARN] Could not fetch {url} — {e}")
        return None

# -----------------------
# Domain normalization
# -----------------------
def _hostname(url: str) -> str:
    try:
        host = (urlparse(url).hostname or "").lower()
        return host.replace("www.", "")
    except Exception:
        return ""

def _norm_token(s: str) -> str:
    return "".join(ch for ch in (s or "").lower() if ch.isalnum())

def _norm_domain_for_name_match(host: str) -> str:
    return "".join(ch for ch in (host or "") if ch.isalnum())

# -----------------------
# Roster loading
# -----------------------
def load_roster_domains(path: str = MAIN_ROSTER_PATH) -> Dict[str, Set[str]]:
    """
    Load controlled domains from roster, keyed by company name.
    Supports pipe-separated URLs in a single cell: 'domain1.com|domain2.com|domain3.com'
    Also handles single domains without pipes.
    
    Returns: Dict[company_name, Set[domain_names]]
    This ensures each company only has its own domains marked as controlled.
    """
    company_domains: Dict[str, Set[str]] = {}

    if not os.path.exists(path):
        print(f"[WARN] roster not found at {path}; proceeding with empty controlled set")
        return company_domains

    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
        cols = {c.strip().lower(): c for c in df.columns}
        
        # Find company and website columns
        company_col = None
        for key in ["company"]:
            if key in cols:
                company_col = cols[key]
                break
        
        website_col = None
        for key in ["website", "websites", "domain", "url", "site", "homepage"]:
            if key in cols:
                website_col = cols[key]
                break
        
        if not company_col or not website_col:
            print(f"[WARN] Missing company or website column in {path}")
            return company_domains
        
        print(f"[INFO] Loading controlled domains from {path} (company: {company_col}, websites: {website_col})...")
        
        for idx, row in df.iterrows():
            company = str(row[company_col]).strip()
            if not company or company.lower() == "nan":
                continue
            
            val = row[website_col]
            if pd.isna(val):
                continue
            
            val = str(val).strip()
            if not val or val.lower() == "nan":
                continue
            
            # Initialize set for this company if not exists
            if company not in company_domains:
                company_domains[company] = set()
            
            # Split on pipe character to support multiple URLs per company
            # This works for both "apple.com" (single) and "apple.com|support.apple.com" (multiple)
            urls = val.split("|")
            
            for url in urls:
                url = url.strip()
                if not url or url.lower() == "nan":
                    continue
                
                # Add http/https prefix if missing for URL parsing
                if not url.startswith(("http://", "https://")):
                    url = f"http://{url}"
                
                host = _hostname(url)
                if host and "." in host:
                    company_domains[company].add(host)
                    print(f"  ✓ {company}: {host}")
        
        total_domains = sum(len(domains) for domains in company_domains.values())
        print(f"[OK] Loaded {len(company_domains)} companies with {total_domains} total controlled domains")
                        
    except Exception as e:
        print(f"[WARN] failed reading roster at {path}: {e}")

    return company_domains

# -----------------------
# Control classification
# -----------------------
def classify_control(company: str, url: str, company_domains: Dict[str, Set[str]]) -> bool:
    """
    Classify if a URL is controlled by a company using three rules:
    (1) Always-controlled platforms (social media, etc.)
    (2) Domains specific to this company from the roster
    (3) Domain contains the company's brand token
    """
    host = _hostname(url)
    if not host:
        return False

    # Rule 1: Always-controlled social platforms
    for good in ALWAYS_CONTROLLED_DOMAINS:
        if host == good or host.endswith("." + good):
            return True

    # Rule 2: Company-specific roster domains (ONLY this company's domains)
    company_specific_domains = company_domains.get(company, set())
    for rd in company_specific_domains:
        if host == rd or host.endswith("." + rd):
            return True

    # Rule 3: Domain contains the brand token (proper subdomain matching)
    brand_token = _norm_token(company)
    if brand_token:
        # Split hostname into parts and normalize each part
        # For "news.apple.com": parts are ["news", "apple", "com"]
        # For "apple.com": parts are ["apple", "com"]
        host_parts = host.split('.')
        normalized_parts = [_norm_token(part) for part in host_parts if part]
        
        # Check if brand_token matches any part of the domain (except TLD)
        # This prevents false positives like "pineapple.com" matching "apple"
        if brand_token in normalized_parts[:-1]:  # Exclude the TLD (last part)
            return True

    return False

# -----------------------
# Sentiment
# -----------------------
def vader_label_on_title(analyzer: SentimentIntensityAnalyzer, title: str) -> Tuple[float, str]:
    s = analyzer.polarity_scores(title or "")
    c = s.get("compound", 0.0)
    if c >= 0.2:
        lab = "positive"
    elif c <= -0.1:
        lab = "negative"
    else:
        lab = "neutral"
    return c, lab

# -----------------------
# Main processing
# -----------------------
def process_for_date(target_date: str) -> None:
    print(f"[INFO] Processing brand SERPs for {target_date} …")
    ensure_dirs()

    company_domains = load_roster_domains()

    url = S3_URL_TEMPLATE.format(date=target_date)
    raw = fetch_csv_from_s3(url)
    if raw is None or raw.empty:
        print(f"[WARN] No raw brand SERP data available for {target_date}. Nothing to write.")
        return

    expected = ["company", "position", "title", "link", "snippet"]
    for col in expected:
        if col not in raw.columns:
            raw[col] = ""

    analyzer = SentimentIntensityAnalyzer()

    processed_rows = []
    for _, row in raw.iterrows():
        company = str(row.get("company", "") or "").strip()
        if not company:
            continue

        title = str(row.get("title", "") or "").strip()
        url = str(row.get("link", "") or "").strip()
        snippet = str(row.get("snippet", "") or "").strip()

        pos_val = row.get("position", 0)
        try:
            position = int(float(pos_val) if pos_val not in (None, "") else 0)
        except Exception:
            position = 0

        controlled = classify_control(company, url, company_domains)

        # --- sentiment rules (deterministic order) ---
        host = _hostname(url)

        # 1) Force negative for reddit.com (and subdomains)
        if host == "reddit.com" or (host and host.endswith(".reddit.com")):
            label = "negative"
        # 2) Force negative for legal-trouble titles (lawsuit, sued, settlement, fines, etc.)
        elif _title_mentions_legal_trouble(title):
            label = "negative"
        else:
            # 3) Neutralize certain brand terms in the title
            if _should_neutralize_title(title):
                label = "neutral"
            else:
                _, label = vader_label_on_title(analyzer, title)

            # 4) Force positive if controlled — but ONLY if we didn't already force negative above
            if FORCE_POSITIVE_IF_CONTROLLED and controlled:
                label = "positive"

        processed_rows.append({
            "date": target_date,
            "company": company,
            "title": title,
            "url": url,
            "position": position,
            "snippet": snippet,
            "sentiment": label,
            "controlled": controlled,
        })

    if not processed_rows:
        print(f"[WARN] No processed rows for {target_date}.")
        return

    rows_df = pd.DataFrame(processed_rows)
    row_out_path = os.path.join(OUT_ROWS_DIR, f"{target_date}-brand-serps-modal.csv")
    rows_df.to_csv(row_out_path, index=False)
    print(f"[OK] Wrote row-level SERPs → {row_out_path}")

    agg = (
        rows_df.groupby("company", as_index=False)
        .agg(
            total=("company", "size"),
            controlled=("controlled", "sum"),
            negative_serp=("sentiment", lambda s: (s == "negative").sum()),
            neutral_serp=("sentiment", lambda s: (s == "neutral").sum()),
            positive_serp=("sentiment", lambda s: (s == "positive").sum()),
        )
    )
    agg.insert(0, "date", target_date)

    daily_out_path = os.path.join(OUT_DAILY_DIR, f"{target_date}-brand-serps-table.csv")
    agg.to_csv(daily_out_path, index=False)
    print(f"[OK] Wrote daily aggregate → {daily_out_path}")

    if os.path.exists(OUT_ROLLUP):
        roll = pd.read_csv(OUT_ROLLUP)
        roll = roll[roll["date"] != target_date]
        roll = pd.concat([roll, agg], ignore_index=True)
    else:
        roll = agg.copy()

    cols = [
        "date",
        "company",
        "total",
        "controlled",
        "negative_serp",
        "neutral_serp",
        "positive_serp",
    ]
    roll = roll[cols].sort_values(["date", "company"]).reset_index(drop=True)
    roll.to_csv(OUT_ROLLUP, index=False)
    print(f"[OK] Updated rolling index → {OUT_ROLLUP}")

def main() -> None:
    args = parse_args()
    date_str = get_target_date(args.date)
    process_for_date(date_str)

if __name__ == "__main__":
    main()
