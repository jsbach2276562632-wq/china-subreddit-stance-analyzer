import time
import random
from datetime import datetime, timedelta, timezone

import requests
import pandas as pd

# Subs chosen intensionally to maximize discussion about Chinese affairs,
# user can replace these with any other subreddits.
subs = [
    #basic and universal discussion
    "China",
    "Sino",
    "Chinese",
    "AskAChinese",
    "AskChina",
    
    #commercial
    "ChinaStocks",
    "Chinavisa",

    #cultural
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
    url = f"https://www.reddit.com/r/{subreddit}/new.json" #reddit Self registering to the API is stuck since last few months, so using json endpoint
    after = None # means we are at the first page. after means turn to next page
    rows = []
    backoff = 8.0  # set a backoff for rate limit

    while len(rows) < max_posts:
        params = {"limit": 100} #reddit returns only 100 post for each page. to be honest, I asked Chatgpt how to fix this
        if after:
            params["after"] = after 

        r = requests.get(url, headers=HEADERS, params=params, timeout=20)

        print("Requesting:", url, params)
        print("Status:", r.status_code)


        if r.status_code == 429: #429 error alwasys, Chatgpt told me If Reddit tells you how long to wait, respect it
            retry_after = r.headers.get("Retry-After")
            if retry_after is not None:
                wait = float(retry_after)
            else:
                wait = max(backoff, 15.0)

            # so that it wont hit the same limit boundary. this part is taught by chatgpt
            time.sleep(wait + random.uniform(0, 2.0))
            backoff = min(backoff * 1.7, 120)
            continue

        if r.status_code != 200:
            time.sleep(backoff + random.uniform(0, 2.0))
            backoff = min(backoff * 1.7, 120)
            continue

        backoff = 8.0


        #read the data
        data = r.json()
        children = data["data"]["children"]

        for c in children:
            d = c["data"] # d is a dictionary after json
            text = (d.get("title","") + " " + (d.get("selftext") or "")).strip() #if title exist then get it otherwise just space
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

        time.sleep(6.0 + random.uniform(0, 3.0))  # randomly slow down to reduce rate-limit risk again.....

    return rows

all_rows = []

for s in subs:
    all_rows += fetch(s, 1500)

df = pd.DataFrame(all_rows).drop_duplicates(subset=["id"])
df.to_csv("data/raw/raw_reddit.csv", index=False)