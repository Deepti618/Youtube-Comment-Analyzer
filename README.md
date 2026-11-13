# YouTube Comment Sentiment Analyzer

Paste a YouTube link and see how the audience feels: an at-a-glance verdict,
a mood bar, the most liked positive and negative comments, common words, and a
searchable table of every comment with CSV export.

## Setup

```bash
pip install -r requirements.txt
```

Get a free API key: Google Cloud Console -> enable **YouTube Data API v3** ->
Credentials -> Create API key.

Provide it in ONE of these ways (never hard-code it, never commit it):

- `.streamlit/secrets.toml` -> `YOUTUBE_API_KEY = "..."` (copy the `.example` file)
- `.env` file -> `YOUTUBE_API_KEY=...` (copy `.env.example`)
- environment variable `YOUTUBE_API_KEY`

## Run

```bash
streamlit run app.py        # web app
python fetch_comments.py    # command-line version
```

## Using the app

- **Settings sidebar:** choose how many comments (50-500) and whether to use top
  or newest comments. The *Neutral zone* slider re-labels results instantly.
- **Tabs:** Highlights, Common words (all / positive / negative), All comments
  (filter, search, sort, download).

## Files

| File | Purpose |
|------|---------|
| `app.py` | Streamlit UI |
| `fetch_comments.py` | URL parsing, video info, paginated comment fetching, CLI |
| `sentiment.py` | TextBlob sentiment logic |
| `.streamlit/config.toml` | Theme |

## Limits

TextBlob is lexicon-based, so sarcasm, slang and emoji are often misread. Only
top-level comments are analyzed (not replies). The free API quota is 10,000
units per day; each analysis uses roughly 1 unit per 100 comments plus 1.
