import html
import os

import pandas as pd
import streamlit as st
from wordcloud import STOPWORDS, WordCloud

from fetch_comments import (
    CommentFetchError,
    extract_video_id,
    get_comments,
    get_video_info,
)
from sentiment import THRESHOLD, get_polarity, label_from_polarity

st.set_page_config(
    page_title="YouTube Comment Sentiment Analyzer",
    page_icon="💬",
    layout="centered",
)

COLORS = {"Positive": "#17806F", "Neutral": "#C3CAD6", "Negative": "#C23B4F"}
EXTRA_STOPWORDS = {
    "video", "one", "like", "just", "really", "will", "get", "got", "much",
    "also", "even", "amp", "https", "www", "com", "youtube", "im", "dont",
}
STOP = STOPWORDS | EXTRA_STOPWORDS

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,600;12..96,800&family=DM+Sans:wght@400;500;700&display=swap');
:root{--ink:#1B2430;--muted:#5B6678;--line:#E2E6EE;--pos:#17806F;--neu:#C3CAD6;--neg:#C23B4F;}
.stApp{font-family:'DM Sans',system-ui,sans-serif;color:var(--ink);}
.block-container{max-width:980px;padding-top:2.5rem;padding-bottom:4rem;}
header[data-testid="stHeader"]{background:transparent;}
footer{visibility:hidden;}
h1,h2,h3,.verdict,.empty h2{font-family:'Bricolage Grotesque','DM Sans',sans-serif;letter-spacing:-0.015em;}
h1{font-weight:800;font-size:2.4rem;line-height:1.1;margin-bottom:.25rem;}
.lede{color:var(--muted);font-size:1.05rem;margin:0 0 1.5rem;max-width:60ch;}

[data-testid="stForm"]{background:#fff;border:1px solid var(--line);border-radius:14px;padding:1.1rem 1.25rem;}
[data-testid="stForm"] button{background:var(--ink);color:#fff;border:0;border-radius:10px;font-weight:700;min-height:2.6rem;}
[data-testid="stForm"] button:hover{background:#2C3A4D;color:#fff;}
[data-testid="stForm"] button:focus-visible{outline:3px solid #7aa7ff;outline-offset:2px;}

.card{background:#fff;border:1px solid var(--line);border-radius:14px;padding:1.4rem 1.5rem;margin:1rem 0 1.5rem;}
.video{display:flex;gap:1rem;align-items:center;margin-top:1.5rem;}
.video img{width:136px;border-radius:8px;flex:none;}
.vtitle{font-weight:700;font-size:1.05rem;line-height:1.3;}
.vmeta{color:var(--muted);font-size:.9rem;}

.verdict{font-size:2.4rem;font-weight:800;line-height:1.1;}
.sub{color:var(--muted);margin:.4rem 0 1.1rem;}
.bar{display:flex;gap:3px;height:52px;border-radius:12px;overflow:hidden;}
.seg{display:flex;align-items:center;justify-content:center;font-weight:700;color:#fff;min-width:4px;}
.seg.positive{background:var(--pos);} .seg.negative{background:var(--neg);}
.seg.neutral{background:var(--neu);color:var(--ink);}
.legend{display:flex;flex-wrap:wrap;gap:1.4rem;margin-top:.9rem;font-size:.95rem;}
.key{display:inline-flex;align-items:center;gap:.45rem;}
.sw{width:12px;height:12px;border-radius:3px;display:inline-block;}
.sw.positive{background:var(--pos);} .sw.negative{background:var(--neg);} .sw.neutral{background:var(--neu);}

.quote{background:#fff;border:1px solid var(--line);border-left:5px solid var(--neu);border-radius:10px;padding:.85rem 1rem;margin-bottom:.7rem;line-height:1.5;overflow-wrap:anywhere;}
.quote.positive{border-left-color:var(--pos);} .quote.negative{border-left-color:var(--neg);}
.qmeta{color:var(--muted);font-size:.82rem;margin-top:.4rem;}

.empty{background:#fff;border:1px dashed #C9CFDB;border-radius:14px;padding:2rem 1.75rem;margin-top:1.25rem;}
.empty h2{font-size:1.5rem;margin:0 0 .4rem;}
.empty p{color:var(--muted);margin:.25rem 0;}
.note{color:var(--muted);font-size:.85rem;margin-top:1.5rem;}

button[role="tab"]{font-weight:700;}
@media (max-width:640px){.verdict{font-size:1.8rem;}.video img{width:96px;}.bar{height:44px;}}
@media (prefers-reduced-motion:reduce){*{transition:none!important;animation:none!important;}}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ---------- helpers ----------
def load_api_key():
    try:
        if "YOUTUBE_API_KEY" in st.secrets:
            return st.secrets["YOUTUBE_API_KEY"]
    except Exception:
        pass
    return os.getenv("YOUTUBE_API_KEY")


@st.cache_data(show_spinner=False, ttl=900)
def load(video_id, max_comments, order, api_key):
    info = get_video_info(video_id, api_key)
    df = pd.DataFrame(get_comments(video_id, api_key, max_comments, order))
    if not df.empty:
        df["polarity"] = df["text"].map(get_polarity)
    return info, df


@st.cache_data(show_spinner=False)
def cloud(text, color):
    wc = WordCloud(
        width=1000, height=420, background_color="white", stopwords=STOP,
        color_func=lambda *a, **k: color, max_words=80, collocations=False,
    ).generate(text)
    return wc.to_array()


def clean(text, limit=280):
    short = text if len(text) <= limit else text[:limit].rstrip() + "…"
    return html.escape(short).replace("\n", "<br>")


def quote(row, label):
    return (
        f'<div class="quote {label.lower()}">{clean(row.text)}'
        f'<div class="qmeta">by {html.escape(row.author)}, {row.likes:,} likes</div></div>'
    )


def verdict(pct):
    pos, neg, neu = pct["Positive"], pct["Negative"], pct["Neutral"]
    if neu >= 50:
        return "Mostly neutral"
    if abs(pos - neg) < 10:
        return "Mixed reactions"
    return "Mostly positive" if pos > neg else "Mostly negative"


def mood_bar(counts, total):
    segs = ""
    for label in ("Negative", "Neutral", "Positive"):
        n = counts[label]
        if n:
            pct = n / total * 100
            text = f"{pct:.0f}%" if pct >= 7 else ""
            segs += f'<div class="seg {label.lower()}" style="flex:{n}">{text}</div>'
    legend = "".join(
        f'<span class="key"><i class="sw {l.lower()}"></i>{l} <b>{counts[l]:,}</b></span>'
        for l in ("Positive", "Neutral", "Negative")
    )
    aria = ", ".join(f"{counts[l]} {l.lower()}" for l in ("Positive", "Neutral", "Negative"))
    return (
        f'<div class="bar" role="img" aria-label="Comment sentiment: {aria}">{segs}</div>'
        f'<div class="legend">{legend}</div>'
    )


# ---------- header ----------
st.title("YouTube Comment Sentiment Analyzer")
st.markdown(
    '<p class="lede">Paste a video link to see whether its commenters are '
    "happy, unhappy or indifferent.</p>",
    unsafe_allow_html=True,
)

api_key = load_api_key()
if not api_key:
    st.error(
        "No API key found. Add `YOUTUBE_API_KEY` to `.streamlit/secrets.toml` "
        "or set it as an environment variable, then restart the app."
    )
    st.stop()

# ---------- sidebar settings ----------
with st.sidebar:
    st.header("Settings")
    max_comments = st.slider("Comments to analyze", 50, 500, 200, step=50)
    order_label = st.selectbox("Which comments", ["Top comments", "Newest first"])
    st.caption("These two apply the next time you click Analyze.")
    threshold = st.slider(
        "Neutral zone", 0.0, 0.3, THRESHOLD, step=0.01,
        help="Comments scoring close to zero are called neutral. "
             "Raise this to get more neutral results. Updates instantly.",
    )

# ---------- input ----------
with st.form("analyze"):
    col_url, col_btn = st.columns([5, 1.2], vertical_alignment="bottom")
    url = col_url.text_input(
        "YouTube link", placeholder="https://www.youtube.com/watch?v=..."
    )
    submitted = col_btn.form_submit_button("Analyze", width="stretch")

if submitted:
    video_id = extract_video_id(url)
    if video_id is None:
        st.session_state.pop("active", None)
        st.error(
            "That doesn't look like a YouTube video link. Use a link like "
            "youtube.com/watch?v=... or youtu.be/..."
        )
        st.stop()
    st.session_state["active"] = {
        "video_id": video_id,
        "max_comments": max_comments,
        "order": "time" if order_label == "Newest first" else "relevance",
        "order_label": order_label,
    }

active = st.session_state.get("active")

# ---------- empty state ----------
if not active:
    st.markdown(
        '<div class="empty"><h2>See how viewers feel about a video</h2>'
        "<p>Paste any public YouTube link above and click Analyze.</p>"
        "<p>Watch links, youtu.be links and Shorts links all work.</p></div>",
        unsafe_allow_html=True,
    )
    st.stop()

# ---------- load data ----------
try:
    with st.spinner("Reading comments..."):
        info, df = load(
            active["video_id"], active["max_comments"], active["order"], api_key
        )
except CommentFetchError as e:
    st.error(str(e))
    st.stop()

if df.empty:
    st.warning("This video has no comments yet, so there is nothing to analyze.")
    st.stop()

df["sentiment"] = df["polarity"].map(lambda p: label_from_polarity(p, threshold))
total = len(df)
counts = {l: int((df["sentiment"] == l).sum()) for l in COLORS}
pct = {l: counts[l] / total * 100 for l in COLORS}

# ---------- results ----------
if info["thumbnail"]:
    cc = (
        f'{info["comment_count"]:,} comments on this video'
        if info["comment_count"] is not None else ""
    )
    st.markdown(
        f'<div class="video"><img src="{html.escape(info["thumbnail"])}" alt="">'
        f'<div><div class="vtitle">{html.escape(info["title"])}</div>'
        f'<div class="vmeta">{html.escape(info["channel"])}</div>'
        f'<div class="vmeta">{cc}</div></div></div>',
        unsafe_allow_html=True,
    )

st.markdown(
    f'<div class="card"><div class="verdict">{verdict(pct)}</div>'
    f'<p class="sub">{pct["Positive"]:.0f}% positive, {pct["Negative"]:.0f}% negative '
    f'and {pct["Neutral"]:.0f}% neutral across {total:,} comments.</p>'
    f"{mood_bar(counts, total)}</div>",
    unsafe_allow_html=True,
)

tab_high, tab_words, tab_all = st.tabs(["Highlights", "Common words", "All comments"])

with tab_high:
    left, right = st.columns(2)
    for col, label, asc in ((left, "Positive", False), (right, "Negative", True)):
        col.subheader(f"Most liked {label.lower()}")
        sub = df[df["sentiment"] == label].sort_values(
            ["likes", "polarity"], ascending=[False, asc]
        ).head(3)
        if sub.empty:
            col.caption(f"No {label.lower()} comments in this set.")
        for row in sub.itertuples():
            col.markdown(quote(row, label), unsafe_allow_html=True)

with tab_words:
    pick = st.radio("Show words from", ["All", "Positive", "Negative"], horizontal=True)
    subset = df if pick == "All" else df[df["sentiment"] == pick]
    text = " ".join(subset["text"])
    color = "#1B2430" if pick == "All" else COLORS[pick]
    if len(subset) < 3 or len(text.split()) < 10:
        st.info(f"Not enough {pick.lower()} comments to build a word cloud.")
    else:
        st.image(cloud(text, color))

with tab_all:
    f1, f2, f3 = st.columns([2, 2, 1.6])
    chosen = f1.multiselect("Sentiment", list(COLORS), default=list(COLORS))
    query = f2.text_input("Search comments", placeholder="Type a word")
    sort = f3.selectbox("Sort by", ["Most liked", "Most positive", "Most negative", "Newest"])

    view = df[df["sentiment"].isin(chosen)]
    if query:
        view = view[view["text"].str.contains(query, case=False, regex=False)]
    sort_map = {
        "Most liked": ("likes", False), "Most positive": ("polarity", False),
        "Most negative": ("polarity", True), "Newest": ("published", False),
    }
    col_name, asc = sort_map[sort]
    view = view.sort_values(col_name, ascending=asc)

    st.caption(f"Showing {len(view):,} of {total:,} comments")
    table = view.rename(columns={
        "text": "Comment", "author": "Author", "likes": "Likes",
        "published": "Date", "sentiment": "Sentiment", "polarity": "Score",
    })[["Comment", "Sentiment", "Score", "Likes", "Author", "Date"]]
    st.dataframe(
        table, hide_index=True, width="stretch",
        column_config={
            "Comment": st.column_config.TextColumn(width="large"),
            "Score": st.column_config.NumberColumn(format="%.2f"),
        },
    )
    st.download_button(
        "Download as CSV", table.to_csv(index=False).encode("utf-8"),
        file_name=f'{active["video_id"]}_sentiment.csv', mime="text/csv",
    )

st.markdown(
    f'<p class="note">Based on {total:,} top-level comments ({active["order_label"].lower()}). '
    "Scores come from TextBlob, which can misread sarcasm, slang and emoji.</p>",
    unsafe_allow_html=True,
)
