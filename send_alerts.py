#!/usr/bin/env python3
"""
Reads the latest daily_counts.csv files for each entity type (brands, CEOs, ROCs)
and triggers email alerts if the negative sentiment threshold is met.

This script is intended to be run after the main data processing scripts
(news_sentiment.py, news_sentiment_ceos.py, news_sentiment_roc.py) have completed.
"""

import os
import pandas as pd
from email_utils import check_and_send_alerts

# --- Configuration ---
# Map entity types to their data file and the name of the entity column in that file.
COUNTS_CONFIG = {
    "Brand": {"file": "data/daily_counts.csv", "col": "brand"},
    "CEO": {"file": "data_ceos/daily_counts.csv", "col": "ceo"},
    "ROC": {"file": "data_roc/daily_counts.csv", "col": "brand"}, 
}

def main():
    """
    Main function to read data and trigger alerts for each entity type.
    """
    print("=== Starting Alert Check ===")

    KIT_API_KEY = os.environ.get("KIT_API_KEY")
    KIT_TAG_ID = os.environ.get("KIT_TAG_ID")

    if not (KIT_API_KEY and KIT_TAG_ID):
        raise SystemExit("Error: KIT_API_KEY and KIT_TAG_ID environment variables are not set.")

    for entity_type, config in COUNTS_CONFIG.items():
        file_path = config["file"]
        entity_col = config["col"]

        print(f"\n--- Checking {entity_type} alerts from {file_path} ---")
        if not os.path.exists(file_path):
            print("Warning: Counts file not found, skipping.")
            continue

        try:
            df = pd.read_csv(file_path)
            if df.empty:
                print("Counts file is empty, skipping.")
                continue

            # Ensure date column is sorted to find the most recent date
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values(by='date', ascending=False)
            
            most_recent_date = df['date'].iloc[0].strftime('%Y-%m-%d')
            print(f"Latest data is from: {most_recent_date}")

            # Filter for the most recent date
            latest_counts_df = df[df['date'] == most_recent_date].copy()

            # Rename the specific entity column to 'brand' for the generic alert function
            latest_counts_df.rename(columns={entity_col: 'brand'}, inplace=True)

            counts_for_alerting = latest_counts_df.to_dict("records")

            check_and_send_alerts(
                counts_for_alerting,
                most_recent_date,
                KIT_API_KEY,
                KIT_TAG_ID,
                entity_type=entity_type,
            )

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    print("\n=== Alert Check Complete ===")


if __name__ == "__main__":
    main()
