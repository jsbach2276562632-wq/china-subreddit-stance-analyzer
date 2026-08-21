import pandas as pd
import streamlit as st
import plotly.express as px

@st.cache_data
def load_data():
    svc = pd.read_csv("data/processed/reddit_with_mention_regime_labeled.csv")
    distilbert = pd.read_csv("data/processed/reddit_with_mention_regime_labeled_distilbert.csv")

    for df in [svc, distilbert]:
        df["subreddit"] = df["subreddit"].astype(str)
        df["mention_regime"] = pd.to_numeric(df["mention_regime"], errors="coerce").fillna(0).astype(int)
        df["predicted_label"] = df["predicted_label"].astype(str).str.lower()
        df["mention_regime_or_not"] = df["mention_regime"].map({
            1: "regime mentioned",
            0: "not mentioned"
        })

    return svc, distilbert

path_svc, path_distilbert = load_data()

st.title("Reddit posts' stance toward China Analyzer")
st.caption("Hi! In this study we collected posts from China-related subreddits and labeled them with their stances by training models." \
 "Here you can check the stance distributions under different China-related narrative: namely, are their stances different when  mention the PRC regime?")

model_choice = st.radio("Choose prediction model", ["SVC", "DistilBERT"], horizontal=True)

df = path_svc if model_choice == "SVC" else path_distilbert

all_subs = sorted(df["subreddit"].dropna().unique().tolist())

selected_subs = st.multiselect(
    "Choose subreddit(s)",
    options=all_subs,
    default=all_subs[:4] if len(all_subs) >= 4 else all_subs
)

if not selected_subs:
    st.warning("Please select at least one subreddit.")
    st.stop()

filtered = df[df["subreddit"].isin(selected_subs)].copy()

if filtered.empty:
    st.warning("Please select at least one subreddit.")
    st.stop()

sample_size = (
    filtered.groupby("mention_regime_or_not")
    .size()
    .reset_index(name="n")
)

c1, c2 = st.columns(2)
with c1:
    st.metric("Selected subreddits", len(selected_subs))
with c2:
    st.metric("Total posts", len(filtered))

st.subheader("Sample size by regime mention")
st.dataframe(sample_size, use_container_width=True)

if (sample_size["n"] < 30).any():
    st.warning("Some sample groups have very small sample sizes, espacially when you chosed only a few subreddits. Interpret results with caution.")

plot_df = (
    filtered.groupby(["mention_regime_or_not", "predicted_label"])
    .size()
    .reset_index(name="count")
)

plot_df["percent"] = (
    plot_df.groupby("mention_regime_or_not")["count"]
    .transform(lambda x: x / x.sum() * 100)
)

fig = px.bar(
    plot_df,
    x="predicted_label",
    y="percent",
    color="mention_regime_or_not",
    barmode="group",
    title="Stance distribution by regime mention",
    labels={
        "predicted_label": "Predicted stance",
        "percent": "Percentage of posts",
        "mention_regime_or_not": "regime mention"
    }
)

st.plotly_chart(fig, use_container_width=True)

st.subheader("Summary table")
st.dataframe(plot_df, use_container_width=True)