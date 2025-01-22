import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load and preprocess the dataset
df = pd.read_csv("dataset_phishing.csv")
df['labels'] = df['status'].map({'phishing': 1, 'legitimate': 0})
X_raw = df['url']
y = df['labels']

# Text vectorization
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(X_raw).toarray()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Naive Bayes classifier
model = GaussianNB()
model.fit(X_train, y_train)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    url = data.get('url')
    if not url:
        return jsonify({'error': 'No URL provided'}), 400
    url_vectorized = vectorizer.transform([url]).toarray()
    prediction = model.predict(url_vectorized)
    result = "Phishing Detected" if prediction[0] == 1 else "Safe URL"
    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(debug=True)
