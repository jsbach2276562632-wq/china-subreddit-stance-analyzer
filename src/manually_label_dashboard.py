import pandas as pd
import streamlit as st
import random
import json

from pathlib import Path

df = pd.read_json("data/labeling/label_sample.jsonl", lines=True)
history = Path("data/labeling/manually_labeled_sample.json")

st.title("Manual stance annotation")
st.caption(
    "Assign each sampled Reddit post to one of three stance categories: "
    "positive, critical, or neutral."
)

if "tweets" not in st.session_state:
    st.session_state.tweets = df

if "labeled" not in st.session_state:
    if history.exists():
        st.session_state.labeled = json.loads(history.read_text(encoding="utf-8"))
    else:
        st.session_state.labeled = []

if "current_id" not in st.session_state:
    st.session_state.current_id = None

tweets_id = [i for i in st.session_state.tweets["id"].tolist()]
labeled_id = [dic["id"] for dic in st.session_state.labeled]
unlabeled_id = [i for i in tweets_id if str(i) not in set(labeled_id)]  

all_num = len(tweets_id)
done_num = len(labeled_id)
to_label_num = len(unlabeled_id)

st.write(f"Completed: {done_num} | Remaining: {to_label_num} | Total: {all_num}")

if to_label_num == 0:
    st.success("Finally!!! All samples are labeled.")
    st.stop()

if st.session_state.current_id is None or str(st.session_state.current_id) in set(labeled_id):
    st.session_state.current_id = random.choice(unlabeled_id)

tweet_id = st.session_state.current_id

text = st.session_state.tweets.loc[
    st.session_state.tweets["id"].astype(str) == str(tweet_id),
    "text"
].iloc[0]

st.subheader(f"sample_id: {tweet_id}")
st.write(text)

def label(lbl):
    st.session_state.labeled.append({"id": str(tweet_id), "label": lbl})

    history.write_text(
        json.dumps(st.session_state.labeled, ensure_ascii=False),
        encoding="utf-8"
    )

    st.session_state.current_id = None
    st.rerun()

col1, col2, col3 = st.columns(3)

with col1:
    st.button("Positive", on_click=label, args=("positive",), use_container_width=True)
with col2:
    st.button("Critical", on_click=label, args=("critical",), use_container_width=True)
with col3:
    st.button("Neutral", on_click=label, args=("neutral",), use_container_width=True)

if st.button("Save progress"):
    history.write_text(
        json.dumps(st.session_state.labeled, ensure_ascii=False),
        encoding="utf-8"
    )
    st.success("Progress saved.")
