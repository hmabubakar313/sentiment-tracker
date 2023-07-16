from flask import Flask, render_template, jsonify
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import re
import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from textblob import TextBlob

app = Flask(__name__)

def clean_tweet(tweet):
    # Tokenization
    tweet = word_tokenize(tweet)
    # Lowercase
    tweet = [word.lower() for word in tweet]
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tweet = [lemmatizer.lemmatize(word) for word in tweet]
    tweet = [word for word in tweet if word.isalpha()]
    tweet = " ".join(tweet)
    # Remove URLs
    tweet = re.sub(r"http\S+", "", tweet)
    tweet = re.sub(r"@\S+", "", tweet)
    tweet = re.sub(r"[^\w\s]", "", tweet)
    tweet = re.sub(r"\d+", "", tweet)
    # Remove stopwords
    tweet = [word for word in tweet.split() if word not in stopwords.words("english")]
    tweet = " ".join(tweet)
    # Stemming
    stemmer = PorterStemmer()
    tweet = stemmer.stem(tweet)

    return tweet

def get_sentiment(text):
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity

    if polarity > 0.5:
        return 'strongly positive'
    elif polarity > 0.2:
        return 'positive'
    elif polarity > -0.2:
        return 'negative'
    elif polarity > -0.5:
        return 'strongly negative'
    else:
        return 'neutral'

# Load data from xlsx
print('Loading data...')
df = pd.read_excel('TweetDataset.xlsx')
print('Data loaded.')

# Clean text data
df['clean_tweet'] = df['Tweet'].apply(lambda x: clean_tweet(x))
print('Text cleaned.')

# Get sentiment
df['sentiment'] = df['clean_tweet'].apply(lambda x: get_sentiment(x))
print('Sentiment analyzed.')

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(df['clean_tweet'], df['sentiment'], test_size=0.2, random_state=42)

# Vectorize the data
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# SVM classifier with RBF kernel for multiclass classification
model = SVC(kernel='rbf', random_state=0)
model.fit(X_train, y_train)

# Predict sentiment
y_pred = model.predict(X_test)

# Check accuracy
accuracy = accuracy_score(y_test, y_pred)
accuracy_percentage = accuracy * 100
accuracy = accuracy_percentage
print("Accuracy: {:.2f}%".format(accuracy_percentage))

# Generate graph
plt.figure(figsize=(10, 6))
sns.countplot(x='sentiment', data=df)
sns.despine()

# Save the graph to an image file
graph_path = 'static/sentiment_graph.png'
plt.savefig(graph_path)

plt.figure(figsize=(10, 10))
# print user location graph for first 100 tweets
data = df.head(100)
# User location with sentiment
sns.countplot(y='User_location', hue=df['sentiment'], data=data)
#.............
location_graph_path = 'static/location_graph.png'
plt.savefig(location_graph_path, bbox_inches='tight')

if not os.path.exists('static'):
    os.makedirs('static')

@app.route('/show_result')
def show_result():
    return render_template('result.html', graph_path=graph_path, location_graph_path='static/location_graph.png', accuracy=accuracy)

if __name__ == '__main__':
    app.run(debug=True)
