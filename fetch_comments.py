"""YouTube helpers + a command-line version of the analyzer.

Run:  python fetch_comments.py
"""
import os
import re

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

try:  # optional: load YOUTUBE_API_KEY from a .env file
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

_ID_RE = re.compile(r"(?:v=|youtu\.be/|/shorts/|/embed/|/live/)([A-Za-z0-9_-]{11})")


class CommentFetchError(Exception):
    """Raised with a user-friendly message when YouTube data can't be fetched."""


def extract_video_id(url):
    if not url:
        return None
    url = url.strip()
    match = _ID_RE.search(url)
    if match:
        return match.group(1)
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", url):  # bare video id
        return url
    return None


def get_api_key():
    key = os.getenv("YOUTUBE_API_KEY")
    if not key:
        raise CommentFetchError(
            "No API key found. Set the YOUTUBE_API_KEY environment variable "
            "(see README.md)."
        )
    return key


def _client(api_key):
    return build("youtube", "v3", developerKey=api_key or get_api_key())


def _friendly(e):
    text = str(e)
    if "commentsDisabled" in text:
        return CommentFetchError("Comments are turned off for this video.")
    if "videoNotFound" in text or e.resp.status == 404:
        return CommentFetchError(
            "Video not found. Check the link and make sure the video is public."
        )
    if "quotaExceeded" in text:
        return CommentFetchError(
            "The daily YouTube API quota is used up. Try again tomorrow."
        )
    if e.resp.status in (400, 403):
        return CommentFetchError(
            "YouTube rejected the request. Check that your API key is valid "
            "and the YouTube Data API v3 is enabled."
        )
    return CommentFetchError(f"YouTube API error: {e}")


def get_video_info(video_id, api_key=None):
    """Title, channel, thumbnail and total comment count for a video."""
    try:
        resp = _client(api_key).videos().list(
            part="snippet,statistics", id=video_id
        ).execute()
    except HttpError as e:
        raise _friendly(e) from e

    items = resp.get("items")
    if not items:
        raise CommentFetchError(
            "Video not found. Check the link and make sure the video is public."
        )
    snippet = items[0]["snippet"]
    stats = items[0].get("statistics", {})
    thumbs = snippet.get("thumbnails", {})
    thumb = (thumbs.get("medium") or thumbs.get("default") or {}).get("url")
    return {
        "title": snippet.get("title", ""),
        "channel": snippet.get("channelTitle", ""),
        "thumbnail": thumb,
        "comment_count": int(stats["commentCount"]) if "commentCount" in stats else None,
    }


def get_comments(video_id, api_key=None, max_comments=200, order="relevance"):
    """Fetch up to max_comments top-level comments (handles pagination).

    order: "relevance" (top comments) or "time" (newest first).
    Returns a list of dicts: text, author, likes, published.
    """
    youtube = _client(api_key)
    comments, page_token = [], None

    try:
        while len(comments) < max_comments:
            response = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=min(100, max_comments - len(comments)),
                pageToken=page_token,
                order=order,
                textFormat="plainText",
            ).execute()

            for item in response.get("items", []):
                s = item["snippet"]["topLevelComment"]["snippet"]
                comments.append({
                    "text": s["textDisplay"],
                    "author": s.get("authorDisplayName", ""),
                    "likes": int(s.get("likeCount", 0)),
                    "published": s.get("publishedAt", "")[:10],
                })

            page_token = response.get("nextPageToken")
            if not page_token:
                break
    except HttpError as e:
        raise _friendly(e) from e

    return comments


def main():
    import matplotlib.pyplot as plt
    from sentiment import analyze_comments

    video_id = extract_video_id(input("Enter YouTube URL: "))
    if video_id is None:
        print("Invalid YouTube URL")
        return

    try:
        comments = get_comments(video_id)
    except CommentFetchError as e:
        print(e)
        return
    if not comments:
        print("No comments found.")
        return

    _, counts = analyze_comments([c["text"] for c in comments])
    total = len(comments)

    print(f"\nAnalyzed {total} comments")
    for label, n in counts.items():
        print(f"{label:<9}: {n} ({n / total * 100:.2f}%)")

    labels, values = list(counts), list(counts.values())
    colors = ["#17806F", "#C23B4F", "#C3CAD6"]

    plt.figure()
    plt.bar(labels, values, color=colors)
    plt.xlabel("Sentiment")
    plt.ylabel("Number of Comments")
    plt.title("YouTube Comment Sentiment Analysis")

    plt.figure()
    plt.pie(values, labels=labels, autopct="%1.1f%%", colors=colors)
    plt.title("Sentiment Distribution")
    plt.show()


if __name__ == "__main__":
    main()
