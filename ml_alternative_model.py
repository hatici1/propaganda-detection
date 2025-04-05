import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from collections import Counter

# Load preprocessed data
df = pd.read_csv("labeled_fragments.csv")

# ❗ Drop rows with missing or empty text
df = df.dropna(subset=["text"])
df = df[df["text"].str.strip().astype(bool)]

# Encode labels to integers
le = LabelEncoder()
df['label'] = le.fit_transform(df['technique'])


# Show label distribution
print("\n🔍 Label distribution:")
print(Counter(df['technique']))

# Stratified split (to keep label distribution balanced)
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)

# Build a pipeline: TF-IDF + SVM
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1,2), max_df=0.85, min_df=2)),
    ("clf", LinearSVC())
])

# Optional: Try hyperparameter tuning with GridSearchCV
param_grid = {
    "clf__C": [0.1, 1.0, 10],
    "tfidf__max_features": [3000, 5000, None]
}

print("\n🔄 Running GridSearchCV (this may take a few minutes)...")
grid = GridSearchCV(pipeline, param_grid, cv=3, scoring="f1_weighted", n_jobs=-1, verbose=1)
grid.fit(X_train, y_train)

# Evaluate
y_pred = grid.predict(X_test)
print("\n✅ Best Parameters:")
print(grid.best_params_)

print("\n📊 Classification Report (SVM + TF-IDF + GridSearch):")
print(classification_report(y_test, y_pred, target_names=le.classes_))
