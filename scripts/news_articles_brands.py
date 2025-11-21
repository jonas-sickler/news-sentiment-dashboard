#!/usr/bin/env python3
import argparse
import csv, os, re, sys, time, math, urllib.parse, requests
from datetime import datetime, timezone
from pathlib import Path
from bs4 import BeautifulSoup
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Words in titles that should NOT contribute to sentiment (often part of brand names)
NEUTRALIZE_TITLE_TERMS = [
    r"\bgrand\b",
    r"\bdiamond\b",
    r"\bsell\b",
    r"\blow\b",
    r"\bdream\b",
]

NEUTRALIZE_TITLE_RE = re.compile("|".join(NEUTRALIZE_TITLE_TERMS), flags=re.IGNORECASE)

def _strip_neutral_terms(headline: str) -> str:
    """Remove configured neutral terms from a headline so they don't affect sentiment."""
    if not headline:
        return ""
    # Replace them with a space, then normalize whitespace
    cleaned = NEUTRALIZE_TITLE_RE.sub(" ", headline)
    return " ".join(cleaned.split())


# Updated paths to use rosters/main-roster.csv and new output directory
BASE = Path(__file__).parent.parent
MAIN_ROSTER = BASE / "rosters" / "main-roster.csv"
OUT_DIR = BASE / "data" / "processed_articles"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Tunables (env overrides)
MAX_PER_ALIAS = int(os.getenv("ARTICLES_MAX_PER_ALIAS", "50"))

def google_news_rss(q):
    qs = urllib.parse.quote(q)
    return f"https://news.google.com/rss/search?q={qs}&hl=en-US&gl=US&ceid=US:en"

def classify(headline, analyzer):
    # Remove neutral words (e.g., Grand, Diamond, Sell, Low) so they don't skew sentiment
    cleaned = _strip_neutral_terms(headline or "")

    # Fall back to original headline if everything got stripped out
    text_for_sentiment = cleaned if cleaned else (headline or "")

    s = analyzer.polarity_scores(text_for_sentiment)
    c = s["compound"]
    if c >= 0.25:
        return "positive"
    if c <= -0.05:
        return "negative"
    return "neutral"

def fetch_one(brand, analyzer, date, pause=1.2):
    url = google_news_rss(f'"{brand}"')
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "xml")
    out = []
    for item in soup.find_all("item"):
        title = (item.title.text or "").strip()
        link  = (item.link.text  or "").strip()
        try:
            if "url=" in link:
                link = urllib.parse.parse_qs(urllib.parse.urlparse(link).query).get("url", [link])[0]
        except Exception:
            pass
        source = (item.source.text or "").strip() if item.source else ""
        sent   = classify(title, analyzer)
        out.append({
            "company": brand,
            "title": title,
            "url": link,
            "source": source,
            "date": date,
            "sentiment": sent
        })
    time.sleep(pause)  # be respectful
    return out[:MAX_PER_ALIAS]  # cap results

def load_companies_from_roster():
    """Load unique company names from rosters/main-roster.csv"""
    if not MAIN_ROSTER.exists():
        raise FileNotFoundError(f"Main roster not found: {MAIN_ROSTER}")
    
    companies = set()
    with MAIN_ROSTER.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        # Normalize header keys
        headers = {h.strip().lower(): h for h in (reader.fieldnames or [])}
        
        # Look for Company column (case-insensitive)
        company_col = None
        for key in ["company"]:
            if key in headers:
                company_col = headers[key]
                break
        
        if not company_col:
            raise ValueError("No 'Company' column found in main-roster.csv")
        
        for row in reader:
            company = (row.get(company_col) or "").strip()
            if company:
                companies.add(company)
    
    return sorted(companies)

def main():
    parser = argparse.ArgumentParser(description="Fetch brand news articles and analyze sentiment")
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Date to use for the data file (YYYY-MM-DD). Defaults to today."
    )
    args = parser.parse_args()
    
    # Use provided date or default to today
    if args.date:
        date = args.date
    else:
        date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    
    # Set output file path based on date
    out_file = OUT_DIR / f"{date}-brand-articles-modal.csv"
    
    if not MAIN_ROSTER.exists():
        print(f"ERROR: {MAIN_ROSTER} not found", file=sys.stderr)
        sys.exit(1)
    
    brands = load_companies_from_roster()
    print(f"Loaded {len(brands)} companies from {MAIN_ROSTER}")
    print(f"Processing articles for date: {date}")
    
    analyzer = SentimentIntensityAnalyzer()

    rows = []
    for b in brands:
        try:
            rows.extend(fetch_one(b, analyzer, date))
        except Exception as e:
            print(f"[WARN] {b}: {e}", file=sys.stderr)

    with out_file.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["company","title","url","source","date","sentiment"])
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {out_file} ({len(rows)} rows)")

if __name__ == "__main__":
    main()
