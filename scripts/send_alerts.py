#!/usr/bin/env python3
"""Run alerting after pipelines complete."""
from __future__ import annotations

import os
from typing import List, Dict, Any
import pandas as pd

from scripts.email_utils import check_and_send_alerts

def _load_counts(csv_path: str) -> pd.DataFrame | None:
    if not os.path.exists(csv_path):
        print(f"Info: {csv_path} not found; skipping.")
        return None
    try:
        import pandas as pd
        df = pd.read_csv(csv_path)

        # --- normalize column names ---
        # which column holds the entity name?
        for c in ["name", "brand", "company", "ceo",]:
            if c in df.columns:
                df = df.rename(columns={c: "name"})
                break
        # which columns hold negative/total counts?
        for c in ["neg", "negative"]:
            if c in df.columns:
                df = df.rename(columns={c: "neg"})
                break
        for c in ["tot", "total"]:
            if c in df.columns:
                df = df.rename(columns={c: "tot"})
                break

        # validate required columns now that we've normalized
        required = {"date", "name", "neg", "tot"}
        missing = required - set(df.columns)
        if missing:
            print(f"Warning: {csv_path} missing columns after normalization: {sorted(missing)}; skipping.")
            return None

        # coerce types
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
        df["neg"]  = pd.to_numeric(df["neg"], errors="coerce").fillna(0).astype(int)
        df["tot"]  = pd.to_numeric(df["tot"], errors="coerce").fillna(0).astype(int)
        return df
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return None


def _prepare_entities_for_date(df: pd.DataFrame, entity_type: str) -> tuple[List[Dict[str, Any]], str] | None:
    if df.empty:
        return None
    most_recent = df["date"].max()
    cur = df[df["date"] == most_recent].copy()

    if entity_type == "CEO" and {"brand", "ceo"}.issubset(cur.columns):
        # Keep both brand (as name) and ceo
        cur = cur.rename(columns={"brand": "name"})
        cur = cur.groupby(["name", "ceo"], as_index=False).agg({"neg": "sum", "tot": "sum"})
        entities = cur.to_dict("records")
    else:
        # Default behavior
        cur = cur.groupby("name", as_index=False).agg({"neg": "sum", "tot": "sum"})
        entities = cur.to_dict("records")

    return entities, most_recent.isoformat()

def main() -> None:
    MAILGUN_API_KEY = os.environ.get("MAILGUN_API_KEY")
    MAILGUN_DOMAIN = os.environ.get("MAILGUN_DOMAIN")
    MAILGUN_FROM = os.environ.get("MAILGUN_FROM")
    MAILGUN_TO = os.environ.get("MAILGUN_TO")
    MAILGUN_REGION = os.environ.get("MAILGUN_REGION")
    ALERT_SEND_MODE = os.environ.get("ALERT_SEND_MODE", "same_morning")

    if not (MAILGUN_API_KEY and MAILGUN_DOMAIN and MAILGUN_FROM and MAILGUN_TO):
        raise SystemExit("Error: MAILGUN_API_KEY, MAILGUN_DOMAIN, MAILGUN_FROM, and MAILGUN_TO must be set.")

    recipients = [a.strip() for a in MAILGUN_TO.split(",") if a.strip()]

    targets = [
        ("Brand", "data/daily_counts/brand-articles-daily-counts-chart.csv"),
        ("CEO", "data/daily_counts/ceo-articles-daily-counts-chart.csv"),
    ]

    any_sent = False
    for entity_type, csv_path in targets:
        df = _load_counts(csv_path)
        if df is None:
            continue

        prepared = _prepare_entities_for_date(df, entity_type)
        if not prepared:
            print(f"Info: no rows for {csv_path}; skipping.")
            continue

        entities, run_date_str = prepared
        check_and_send_alerts(
            entities, run_date_str,
            MAILGUN_API_KEY, MAILGUN_DOMAIN, MAILGUN_FROM, recipients,
            entity_type=entity_type, region=MAILGUN_REGION, schedule_mode=ALERT_SEND_MODE
        )
        any_sent = True

    if not any_sent:
        print("Nothing to send across configured targets.")

if __name__ == "__main__":
    main()
