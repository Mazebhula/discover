from transformers import pipeline
from sentiments.read_transcript import read_transcript
import os

def load_products(file_path="products/products.txt"):
    """Load products from a text file into a dictionary."""
    products = {}
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Products file not found at {file_path}")
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and ':' in line:  # Ensure the line is not empty and has a key-value pair
                key, value = line.split(':', 1)  # Split on first ':' only
                products[key.strip()] = value.strip()
    return products
def analyze_sentiment_and_suggest(text):
    sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    result = sentiment_analyzer(text)[0]
    label = result["label"]
    score = result["score"]
    
    products = load_products()
    
    text_lower = text.lower()
    if label == "POSITIVE":
        if "sound" in text_lower:
            suggestion = products.get("positive_sound", products["neutral_default"])
        elif "battery" in text_lower:
            suggestion = products.get("positive_battery", products["neutral_default"])
        else:
            suggestion = products["neutral_default"]
    elif label == "NEGATIVE":
        if "sound" in text_lower:
            suggestion = products.get("negative_sound", products["neutral_default"])
        elif "battery" in text_lower:
            suggestion = products.get("negative_battery", products["neutral_default"])
        else:
            suggestion = products["neutral_default"]
    else:
        suggestion = products["neutral_default"]
    
    print(f"Text: {text}")
    print(f"Sentiment: {label} (Confidence: {score:.2f})")
    print(f"Suggested Product: {suggestion}\n")
    
    return label, suggestion