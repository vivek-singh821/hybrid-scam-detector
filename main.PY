import pandas as pd

# Load dataset
df = pd.read_csv("data/spam.csv", encoding="latin-1")

# Keep only required columns
df = df[['v1', 'v2']]

# Rename columns
df.columns = ['label', 'message']

print("Cleaned dataset:")
print(df.head())

print("\nShape:")
print(df.shape)

print("\nLabel counts:")
print(df['label'].value_counts())
# Convert labels to numbers
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

print("\nAfter converting labels:")
print(df.head())
from sklearn.model_selection import train_test_split

# Features (X) and Target (y)
X = df['message']
y = df['label']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training size:", len(X_train))
print("Testing size:", len(X_test))

from sklearn.feature_extraction.text import TfidfVectorizer

# Create TF-IDF object
vectorizer = TfidfVectorizer(
    ngram_range=(1,2),  # use single words + word pairs
    stop_words='english'
)

# Learn vocabulary from training data and transform it
X_train_tfidf = vectorizer.fit_transform(X_train)

# Only transform test data (DO NOT fit again)
X_test_tfidf = vectorizer.transform(X_test)

print("Training matrix shape:", X_train_tfidf.shape)
print("Testing matrix shape:", X_test_tfidf.shape)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Create model
model = LogisticRegression(
    max_iter=1000,
    class_weight='balanced'
)

# Train model
model.fit(X_train_tfidf, y_train)

# Make predictions on test data
y_pred = model.predict(X_test_tfidf)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

print("\nDetailed Report:")
print(classification_report(y_test, y_pred))
import pickle

# Save trained model
pickle.dump(model, open("model.pkl", "wb"))

# Save TF-IDF vectorizer
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))