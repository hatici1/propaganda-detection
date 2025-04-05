import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

# Load the dataset (assumes load_dataset.py was already run and saved to CSV or same DataFrame structure)
df = pd.read_csv("labeled_fragments.csv")  # or load from memory if integrated

# Drop rows with missing fragment text
df.dropna(subset=["text", "technique"], inplace=True)

# Features and labels
X = df["text"]
y = df["technique"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Logistic Regression classifier
clf = LogisticRegression(max_iter=1000, class_weight='balanced')
clf.fit(X_train_vec, y_train)

# Evaluation
y_pred = clf.predict(X_test_vec)
print("\n🧪 Classification Report (Logistic Regression TF-IDF):")
print(classification_report(y_test, y_pred))

# Optional: Save the model for later use
joblib.dump(clf, "baseline_model.joblib")
joblib.dump(vectorizer, "tfidf_vectorizer.joblib")
