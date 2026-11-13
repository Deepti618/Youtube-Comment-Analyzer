from textblob import TextBlob

# Scores between -THRESHOLD and +THRESHOLD count as Neutral.
THRESHOLD = 0.05


def get_polarity(text):
    """TextBlob polarity: -1 (very negative) to +1 (very positive)."""
    return TextBlob(text).sentiment.polarity


def label_from_polarity(polarity, threshold=THRESHOLD):
    if polarity > threshold:
        return "Positive"
    if polarity < -threshold:
        return "Negative"
    return "Neutral"


def analyze_sentiment(comment, threshold=THRESHOLD):
    """Return 'Positive', 'Negative' or 'Neutral' for a single comment."""
    return label_from_polarity(get_polarity(comment), threshold)


def analyze_comments(comments, threshold=THRESHOLD):
    """comments: list of strings. Returns (results, counts)."""
    results = []
    counts = {"Positive": 0, "Negative": 0, "Neutral": 0}
    for text in comments:
        polarity = get_polarity(text)
        label = label_from_polarity(polarity, threshold)
        counts[label] += 1
        results.append((text, label, polarity))
    return results, counts
