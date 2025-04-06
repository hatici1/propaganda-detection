import pandas as pd
import joblib

# Load the sentence-level test fragments generated earlier
df = pd.read_csv("ml_test_fragments.csv")

# Load the trained model and vectorizer
model = joblib.load("baseline_model.joblib")
vectorizer = joblib.load("tfidf_vectorizer.joblib")

# Transform the sentence text using TF-IDF
X_test = vectorizer.transform(df["text"])
y_pred = model.predict(X_test)

df["predicted"] = y_pred

# Build the submission format
submission_rows = []
for _, row in df.iterrows():
    article_id = row["article_id"]
    label = row["predicted"]
    start = int(row["start"])
    end = int(row["end"])
    if end > start:
        submission_rows.append([article_id, label, start, end])

# Save to TSV
submission_df = pd.DataFrame(submission_rows, columns=["article_id", "technique", "start", "end"])
submission_df.to_csv("submission_from_ml.tsv", sep="\t", index=False, header=False)


print(f"✅ Saved ML submission with {len(submission_df)} rows to 'submission_from_ml.tsv'")