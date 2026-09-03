import re
from pathlib import Path
import pandas as pd

df = pd.read_csv("data/raw/raw_reddit.csv")
df["text"] = df["text"].astype(str)

# Operationalize regime mentions through an explicit, reproducible keyword rule.
mention_regime_pattern = re.compile(
    r"""
    (?ix)

    # mainly about the party and the system
    \b(
        ccp|cpc|
        (chinese|china'?s)\s+communist\s+party|
        communist\s+party\s+of\s+china|
        party[-\s]?state|
        one[-\s]?party(\s+state)?|
        marxism|marxist|leninism|mao(ism)?|
        socialism\s+with\s+chinese\s+characteristics
    )\b

    # core politicians and supreme political centre
    |\b(
        xi(\s+jinping)?|
        zhongnanhai|
        politburo|standing\s+committee|
        congress\s+of\s+the\s+communist\s+party
    )\b

    # --- State organs
    |\b(
        state\s+council|
        ministry\s+of\s+state\s+security|mss|
        public\s+security|mps|
        united\s+front|
        propaganda\s+department|
        pla|people'?s\s+liberation\s+army
    )\b

    # regime-issue 
    |\b(
        tiananmen|june\s+4(th)?|
        xinjiang|uyghur(s)?|
        tibet(an)?|
        national\s+security\s+law
    )\b
    """,
    re.IGNORECASE | re.VERBOSE
)

def detect_text(t):
    t = str(t)
    if mention_regime_pattern.search(t):
        return 1
    else:
        return 0

df["mention_regime"] = df["text"].apply(detect_text)
output_path = Path("data/processed/reddit_with_mention_regime.csv")
output_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(output_path, index=False)
print(f"Saved {len(df)} records to {output_path}")
