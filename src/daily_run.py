"""
Runs predict_next_day() and appends the result to data/predictions_history.json.
Called by the GitHub Actions daily_predict workflow.
"""

import json
import os
from datetime import datetime, timezone

import config
from src.predict import predict_next_day


def run():
    print("Fetching latest data and running prediction...")
    result = predict_next_day(force_refresh=True)
    result["run_timestamp"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    history_path = os.path.join(config.DATA_DIR, "predictions_history.json")
    os.makedirs(config.DATA_DIR, exist_ok=True)

    history = []
    if os.path.exists(history_path):
        with open(history_path) as f:
            history = json.load(f)

    # Remove any existing entry for the same date so re-runs don't duplicate
    history = [h for h in history if h.get("as_of_date") != result["as_of_date"]]
    history.append(result)
    history = sorted(history, key=lambda x: x["as_of_date"])

    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"Saved prediction for {result['as_of_date']}:")
    for k, v in result.items():
        print(f"  {k:35s}: {v}")


if __name__ == "__main__":
    run()
