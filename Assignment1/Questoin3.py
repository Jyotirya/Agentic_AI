from transformers import pipeline

# Create a sentiment analysis pipeline
sentiment_analyzer = pipeline("sentiment-analysis")

# Define a list of five short movie reviews
reviews = [
    "The movie was absolutely fantastic with brilliant performances.",
    "I found the film boring and far too long.",
    "The plot was interesting, but the execution was disappointing.",
    "An excellent cinematic experience with stunning visuals.",
    "The story made no sense and the acting was terrible."
]

# Run sentiment analysis on all reviews in one call
results = sentiment_analyzer(reviews)

# Print each review with its predicted label and confidence score
for review, result in zip(reviews, results):
    label = result["label"]
    score = result["score"]
    print("Review:", review)
    print(f"Sentiment: {label} (Confidence: {score:.4f})")