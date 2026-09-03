import time
import random
from pathlib import Path

import requests
import pandas as pd

# Subreddits selected to cover general, commercial, cultural, and historical
# discussions related to China. Replace this list to adapt the pipeline.
subs = [
    # General discussion
    "China",
    "Sino",
    "Chinese",
    "AskAChinese",
    "AskChina",
    
    # Commercial and practical discussion
    "ChinaStocks",
    "Chinavisa",

    # Cultural and historical discussion
    "travelchina",
    "chinalife",
    "ChineseHistory",
    "ChineseLanguage",
    "CDrama"

    
]  


HEADERS = {
    "User-Agent": "icss-final-project/0.1 (reddit public json collector; student project)"
}

def fetch(subreddit, max_posts=1500):
    url = f"https://www.reddit.com/r/{subreddit}/new.json"
    after = None
    rows = []
    backoff = 8.0  # set a backoff for rate limit

    while len(rows) < max_posts:
        params = {"limit": 100}
        if after:
            params["after"] = after 

        r = requests.get(url, headers=HEADERS, params=params, timeout=20)

        print("Requesting:", url, params)
        print("Status:", r.status_code)


        if r.status_code == 429:
            retry_after = r.headers.get("Retry-After")
            if retry_after is not None:
                wait = float(retry_after)
            else:
                wait = max(backoff, 15.0)

            # Add jitter so repeated requests do not hit the same rate-limit boundary.
            time.sleep(wait + random.uniform(0, 2.0))
            backoff = min(backoff * 1.7, 120)
            continue

        if r.status_code != 200:
            time.sleep(backoff + random.uniform(0, 2.0))
            backoff = min(backoff * 1.7, 120)
            continue

        backoff = 8.0


        # Parse one page of Reddit's listing response.
        data = r.json()
        children = data["data"]["children"]

        for c in children:
            d = c["data"]
            text = (d.get("title", "") + " " + (d.get("selftext") or "")).strip()
            rows.append({
                "id": d.get("id"),
                "subreddit": subreddit,
                "created_utc": d.get("created_utc"),
                "text": text
            })
            if len(rows) >= max_posts:
                break

        after = data["data"].get("after")
        if not after:
            break

        time.sleep(6.0 + random.uniform(0, 3.0))

    return rows

all_rows = []

for s in subs:
    all_rows += fetch(s, 1500)

df = pd.DataFrame(all_rows).drop_duplicates(subset=["id"])
output_path = Path("data/raw/raw_reddit.csv")
output_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(output_path, index=False)
print(f"Saved {len(df)} records to {output_path}")
