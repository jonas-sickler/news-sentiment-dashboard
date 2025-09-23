#!/usr/bin/env python3
"""
Daily Google News RSS -> sentiment counts per brand (with alias support),
plus a short daily "theme" for each brand based on NEGATIVE headlines only. 

Outputs (repo-relative):
- data/articles/YYYY-MM-DD.csv  (item-level rows)
- data/daily_counts.csv         (aggregated per brand/day including 'theme')

Inputs (repo root):
- brands.txt    (one brand per line; optional if aliases.csv already covers all)
- aliases.csv   (CSV with columns: brand,alias)  # optional

Housekeeping:
- Deletes article files and daily_counts rows older than RETENTION_DAYS.
"""

import os
import re
import csv
import time
import html
import feedparser
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from urllib.parse import urlparse, parse_qsl, unquote, urlencode

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ------------- Config -----------------
HL   = "en-US"          # Google News interface language
GL   = "US"             # geography
CEID = "US:en"          # country:lang

REQUEST_PAUSE_SEC   = 1.0   # ← gentler to Google News
MAX_ITEMS_PER_BRAND = 40    # ← cap headlines per brand

# Negative-theme extraction
MIN_THEME_FREQ = 2               # min occurrences for a phrase to count
MAX_THEME_WORDS = 10             # cap theme length

# Soft time-shift so very-early runs still write "yesterday"
EASTERN = ZoneInfo("US/Eastern")
SOFT_SHIFT_HOURS = 6             # if before 6am ET, use previous date

# Data retention (days)
RETENTION_DAYS = 365              # ← purge older than this

# Skip press-release wires etc. (you can add more)
BLOCKED_DOMAINS = {
    "www.prnewswire.com",
    "www.businesswire.com",
    "apnews.com/prnewswire",
}

# --------------------------------------

STOPWORDS = set("""
the a an and or but of for to in on at by with from as about after over under into across amid amidst per
this that these those it its their his her they we you our your i
is are was were be been being has have had do does did will would should can could may might must
new update updates report reports reported says say said see sees seen watch market stock shares share price prices
wins gain gains up down news today latest analyst analysts rating
quarter q1 q2 q3 q4 year yrs 2022 2023 2024 2025 2026 usd billion million percent pct vs inc corp co ltd plc
""".split())

analyzer = SentimentIntensityAnalyzer()

def classify_sentiment(text: str) -> str:
    """VADER-based sentiment label."""
    s = analyzer.polarity_scores(text or "")
    comp = s.get("compound", 0.0)
    if comp >= 0.05:
        return "positive"
    if comp <= -0.05:
        return "negative"
    return "neutral"

def tokens(s: str) -> list[str]:
    s = re.sub(r"[^a-z0-9\s-]", " ", (s or "").lower())
    toks = [t for t in s.split() if t and t not in STOPWORDS and len(t) > 2]
    return toks

def top_ngram_from_titles(brand: str, titles: list[str], min_freq: int = MIN_THEME_FREQ) -> str:
    """
    Build a concise theme from a list of titles (negative-only upstream).
    Prefers frequent trigrams/bigrams; falls back to top words; returns 'None' if nothing stands out.
    """
    if not titles:
        return "None"

    brand_toks = set(tokens(brand))
    uni = Counter()
    bi  = Counter()
    tri = Counter()

    for t in titles:
        toks = [w for w in tokens(t) if w not in brand_toks]
        if not toks:
            continue
        uni.update(toks)
        bi.update(" ".join(toks[i:i+2]) for i in range(max(0, len(toks)-1)))
        tri.update(" ".join(toks[i:i+3]) for i in range(max(0, len(toks)-2)))

    # Prefer trigrams, then bigrams (must meet min frequency)
    for bag in (tri, bi):
        if bag:
            phrase, freq = bag.most_common(1)[0]
            if freq >= min_freq:
                return " ".join(phrase.split()[:MAX_THEME_WORDS])

    # Fallback: top 2–4 individual words
    words = [w for w, _ in uni.most_common(4)]
    if not words:
        return "None"
    return " ".join(words[:MAX_THEME_WORDS])

# --------- Inputs: brands & aliases ----------
def read_brands_txt() -> set[str]:
    brands = set()
    path = "brands.txt"
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s and not s.startswith("#"):
                    brands.add(s)
    return brands

def read_aliases_csv() -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = {}
    path = "aliases.csv"
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                brand = (row.get("brand") or "").strip()
                alias = (row.get("alias") or "").strip()
                if brand and alias:
                    mapping.setdefault(brand, []).append(alias)
    return mapping

def merged_brand_map() -> dict[str, list[str]]:
    """
    Returns { brand: [alias1, alias2, ...] }.
    If brands.txt has brands missing from aliases.csv, include them with no aliases.
    """
    base = read_aliases_csv()
    brands_txt = read_brands_txt()
    for b in brands_txt:
        base.setdefault(b, [])
    return base

# --------- Google News helpers (URL-safe) ----------
_NON_ALNUM = re.compile(r"[^A-Za-z0-9-]")

def quoted_term(term: str) -> str:
    """
    Quote the term if it contains spaces OR any non-alphanumeric/hyphen character.
    This ensures tokens like 'P&G' are treated as a single phrase.
    """
    t = (term or "").strip()
    if not t:
        return t
    if " " in t or _NON_ALNUM.search(t):
        return f'"{t}"'
    return t

def build_query(brand: str, aliases: list[str]) -> str:
    terms = [quoted_term(brand)] + [quoted_term(a) for a in aliases]
    return "(" + " OR ".join([t for t in terms if t]) + ")"

def google_news_rss_url(query: str) -> str:
    """Build a fully URL-encoded Google News RSS URL."""
    params = {"q": query, "hl": HL, "gl": GL, "ceid": CEID}
    return "https://news.google.com/rss/search?" + urlencode(params)

def extract_final_url(url: str) -> str:
    """
    Google News often wraps real article URLs. Try to unwrap:
    - https://news.google.com/articles/... -> check 'url=' or 'u=' params
    - Otherwise return as-is
    """
    try:
        parsed = urlparse(url)
        host = parsed.netloc.lower()
        if host.endswith("news.google.com"):
            qs = dict(parse_qsl(parsed.query))
            for key in ("url", "u"):
                if key in qs and qs[key]:
                    return unquote(qs[key])
    except Exception:
        pass
    return url

def domain_from_url(url: str) -> str:
    try:
        host = urlparse(url).netloc.lower()
        return host
    except Exception:
        return ""

# -------------- Run --------------
def now_eastern_date_str() -> str:
    now = datetime.now(EASTERN)
    if now.hour < SOFT_SHIFT_HOURS:
        now = now - timedelta(hours=SOFT_SHIFT_HOURS)
    return now.date().isoformat()

def fetch_brand_articles(for_date: str, brand: str, aliases: list[str]) -> list[dict]:
    """
    Query Google News RSS for (brand OR aliases).
    Return list of dicts with: date, brand, alias_matched, title, url, domain, sentiment, published
    """
    q = build_query(brand, aliases)
    feed_url = google_news_rss_url(q)
    d = feedparser.parse(feed_url)

    out: list[dict] = []
    if getattr(d, "entries", None) is None:
        return out

    alias_lc = [a.lower() for a in aliases]
    brand_lc = brand.lower()

    for e in d.entries[:MAX_ITEMS_PER_BRAND]:   # ← cap to 40
        title = html.unescape(getattr(e, "title", "") or "")
        link = getattr(e, "link", "") or ""
        link = extract_final_url(link)
        dom = domain_from_url(link)

        if not title or not link:
            continue
        if dom in BLOCKED_DOMAINS:
            continue

        t_lc = title.lower()
        matched = ""
        for a in alias_lc:
            if a and a in t_lc:
                matched = a
                break
        if not matched and brand_lc in t_lc:
            matched = brand

        published = getattr(e, "published", "") or getattr(e, "updated", "") or ""
        sent = classify_sentiment(title)

        out.append({
            "date": for_date,
            "brand": brand,
            "alias_matched": matched,
            "title": title,
            "url": link,
            "domain": dom,
            "sentiment": sent,
            "published": published,
        })

    time.sleep(REQUEST_PAUSE_SEC)  # ← slower pace
    return out

def write_articles_csv(for_date: str, rows: list[dict]) -> None:
    os.makedirs("data/articles", exist_ok=True)
    path = os.path.join("data", "articles", f"{for_date}.csv")
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date","brand","alias_matched","title","url","domain","sentiment","published"])
        for r in rows:
            w.writerow([
                r.get("date",""),
                r.get("brand",""),
                r.get("alias_matched",""),
                r.get("title",""),
                r.get("url",""),
                r.get("domain",""),
                r.get("sentiment",""),
                r.get("published",""),
            ])

def compute_counts_and_themes(for_date: str, article_rows: list[dict]) -> list[dict]:
    """
    Aggregate per brand for 'for_date'.
    Theme is computed from NEGATIVE-only titles per brand.
    Returns list of dicts with: date, brand, total, positive, neutral, negative, theme
    """
    by_brand = defaultdict(list)
    for r in article_rows:
        if (r.get("date") or "") != for_date:
            continue
        b = (r.get("brand") or "").strip()
        if not b:
            continue
        by_brand[b].append(r)

    out_rows: list[dict] = []
    for brand, items in by_brand.items():
        pos = sum(1 for x in items if (x.get("sentiment") or "") == "positive")
        neu = sum(1 for x in items if (x.get("sentiment") or "") == "neutral")
        neg = sum(1 for x in items if (x.get("sentiment") or "") == "negative")
        tot = pos + neu + neg

        negative_titles = [x.get("title","") for x in items if (x.get("sentiment") or "") == "negative"]
        theme = top_ngram_from_titles(brand, negative_titles, MIN_THEME_FREQ)
        if not theme or theme.strip().lower() == "none":
            theme = "None"

        out_rows.append({
            "date": for_date,
            "brand": brand,
            "total": tot,
            "positive": pos,
            "neutral": neu,
            "negative": neg,
            "theme": theme,
        })

    out_rows.sort(key=lambda r: (-int(r["negative"]), r["brand"].lower()))
    return out_rows

def update_daily_counts_csv(new_rows: list[dict]) -> None:
    """
    Upsert: remove existing rows for 'date' then append new_rows.
    """
    os.makedirs("data", exist_ok=True)
    path = os.path.join("data", "daily_counts.csv")
    date = new_rows[0]["date"] if new_rows else None

    existing: list[dict] = []
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if date and (row.get("date") == date):
                    continue
                existing.append(row)

    with open(path, "w", encoding="utf-8", newline="") as f:
        fieldnames = ["date","brand","total","positive","neutral","negative","theme"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

        existing.sort(key=lambda r: (r.get("date",""), r.get("brand","").lower()))
        for r in existing:
            w.writerow({
                "date": r.get("date",""),
                "brand": r.get("brand",""),
                "total": r.get("total",""),
                "positive": r.get("positive",""),
                "neutral": r.get("neutral",""),
                "negative": r.get("negative",""),
                "theme": r.get("theme",""),
            })

        for r in new_rows:
            w.writerow(r)

def purge_old_data(retention_days: int = RETENTION_DAYS) -> None:
    """
    Delete article CSVs and prune daily_counts rows older than the cutoff date.
    """
    cutoff = (datetime.now(EASTERN).date() - timedelta(days=retention_days)).isoformat()

    # Purge article files older than cutoff
    articles_dir = os.path.join("data", "articles")
    if os.path.isdir(articles_dir):
        for name in os.listdir(articles_dir):
            if not name.endswith(".csv"):
                continue
            date_part = name[:-4]  # strip .csv
            # Expect YYYY-MM-DD
            if len(date_part) == 10 and date_part <= cutoff:
                try:
                    os.remove(os.path.join(articles_dir, name))
                    print(f"Purged articles file: {name}")
                except Exception as e:
                    print(f"Warning: could not delete {name}: {e}")

    # Prune daily_counts.csv
    path = os.path.join("data", "daily_counts.csv")
    if not os.path.exists(path):
        return

    kept: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            d = (row.get("date") or "")
            if d >= cutoff:
                kept.append(row)

    with open(path, "w", encoding="utf-8", newline="") as f:
        fieldnames = ["date","brand","total","positive","neutral","negative","theme"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in kept:
            w.writerow({
                "date": r.get("date",""),
                "brand": r.get("brand",""),
                "total": r.get("total",""),
                "positive": r.get("positive",""),
                "neutral": r.get("neutral",""),
                "negative": r.get("negative",""),
                "theme": r.get("theme",""),
            })
    print(f"Purged daily_counts rows older than {cutoff}. Kept rows: {len(kept)}")


def main():
    # Which date are we writing under?
    run_date = os.environ.get("NEWS_DATE") or now_eastern_date_str()
    print(f"Running sentiment for {run_date} (US/Eastern)")

    brand_map = merged_brand_map()  # {brand: [aliases]}
    if not brand_map:
        raise SystemExit("No brands found. Add brands.txt and/or aliases.csv.")

    all_articles: list[dict] = []
    for brand, aliases in brand_map.items():
        rows = fetch_brand_articles(run_date, brand, aliases)
        all_articles.extend(rows)

    # Persist per-article file
    write_articles_csv(run_date, all_articles)

    # Aggregate & compute negative-only themes
    counts = compute_counts_and_themes(run_date, all_articles)

    # Upsert into daily_counts.csv
    update_daily_counts_csv(counts if counts else [])

    # Purge old data
    purge_old_data(RETENTION_DAYS)

    print(f"Done. Brands processed: {len(set(r['brand'] for r in counts)) if counts else 0}. Articles: {len(all_articles)}")

if __name__ == "__main__":
    main()