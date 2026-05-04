"""
Social media platforms generate vast amounts of unstructured data such as tweets, comments, and posts, which are often noisy and not directly usable for analysis. Data collection is the first step in social media analytics, involving gathering public textual data while dealing with challenges like API restrictions, rate limits, and ethical considerations. Tools like Snscrape simplify this process by allowing data extraction without API authentication and supporting advanced queries, making it useful for collecting historical and real-time data for analysis.

Once collected, the data must be preprocessed to make it suitable for analysis. Preprocessing removes noise and standardizes text, improving the accuracy of tasks like sentiment analysis and topic modeling. Common techniques include lowercasing text, removing URLs, HTML tags, punctuation, and stopwords, as well as applying stemming and lemmatization to reduce words to their base forms. Sentiment analysis is then performed to classify text as positive, negative, or neutral using a simple lexicon-based approach, though it has limitations such as inability to detect context, sarcasm, or negations.

After processing, the cleaned data is stored in MongoDB, a NoSQL database that uses flexible JSON-like documents, making it ideal for handling semi-structured social media data. MongoDB offers scalability, flexible querying, and easy integration with Python tools. The stored data can then be used for various business applications, including complaint detection, customer satisfaction monitoring, brand reputation analysis, competitor comparison, and product feedback evaluation."""

# Install required libraries:
# pip install snscrape pandas pymongo nltk

import snscrape.modules.twitter as sntwitter
import pandas as pd
import re
from pymongo import MongoClient
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# डाउनलोड (पहली बार ही जरूरी)
nltk.download('stopwords')
nltk.download('wordnet')

# -------------------------------
# 1. Data Collection (Snscrape)
# -------------------------------
query = "iphone OR android lang:en"
tweets_data = []

for i, tweet in enumerate(sntwitter.TwitterSearchScraper(query).get_items()):
    if i >= 200:  # limit
        break
    tweets_data.append({
        "date": tweet.date,
        "username": tweet.user.username,
        "raw_text": tweet.content
    })

df = pd.DataFrame(tweets_data)

# -------------------------------
# 2. Data Preprocessing
# -------------------------------
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)  # remove URLs
    text = re.sub(r"@\w+|#\w+", "", text)  # remove mentions & hashtags
    text = re.sub(r"[^a-z\s]", "", text)  # remove special chars
    
    words = text.split()
    words = [word for word in words if word not in stop_words]  # remove stopwords
    words = [lemmatizer.lemmatize(word) for word in words]  # lemmatization
    
    return " ".join(words)

df["clean_text"] = df["raw_text"].apply(preprocess)

# -------------------------------
# 3. Sentiment Analysis
# -------------------------------
positive_words = ["good", "great", "love", "excellent", "amazing", "awesome", "best"]
negative_words = ["bad", "poor", "hate", "worst", "terrible", "awful", "disappointing"]

def analyze_sentiment(text):
    words = text.split()
    pos = sum(word in positive_words for word in words)
    neg = sum(word in negative_words for word in words)
    
    if pos > neg:
        return "Positive"
    elif neg > pos:
        return "Negative"
    else:
        return "Neutral"

df["sentiment"] = df["clean_text"].apply(analyze_sentiment)

# -------------------------------
# 4. MongoDB Storage
# -------------------------------
client = MongoClient("mongodb://localhost:27017/")
db = client["social_media_analytics"]
collection = db["tweets_data"]

collection.insert_many(df.to_dict("records"))

print("Data stored in MongoDB successfully!")

# -------------------------------
# 5. Basic Analytics Output
# -------------------------------
print("\nSample Data:")
print(df.head())

print("\nSentiment Distribution:")
print(df["sentiment"].value_counts())