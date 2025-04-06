import os
import pandas as pd

test_dir = "test"
test_articles = []

for fname in os.listdir(test_dir):
    if fname.endswith(".txt"):
        article_id = fname.replace("article", "").replace(".txt", "")
        with open(os.path.join(test_dir, fname), encoding="utf-8") as f:
            full_text = f.read()
        start = 0
        for sent in full_text.split("."):
            sent = sent.strip()
            if sent:
                end = start + len(sent)
                test_articles.append([article_id, start, end, sent, full_text])
                start = end + 1

df = pd.DataFrame(test_articles, columns=["article_id", "start", "end", "text", "full_text"])
df.to_csv("ml_test_fragments.csv", index=False)
print("✅ Saved ml_test_fragments.csv")