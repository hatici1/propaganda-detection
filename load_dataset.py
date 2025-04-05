import os
import glob
import pandas as pd

def load_article(article_path):
    with open(article_path, 'r', encoding='utf-8') as f:
        return f.read()

def load_labels(label_path):
    cols = ['article_id', 'technique', 'start_char', 'end_char']
    return pd.read_csv(label_path, sep='\t', names=cols)

def get_matched_fragments(text, labels_df):
    fragments = []
    for _, row in labels_df.iterrows():
        try:
            span = text[row.start_char:row.end_char]
            fragments.append({
                "article_id": row.article_id,
                "technique": row.technique,
                "start": row.start_char,
                "end": row.end_char,
                "text": span,
                "full_text": text
            })
        except Exception as e:
            print(f"⚠️ Error extracting span for article {row.article_id}: {e}")
    return fragments

def load_all_articles(folder_path):
    all_fragments = []
    label_files = sorted(glob.glob(os.path.join(folder_path, "*.labels.tsv")))

    for label_path in label_files:
        article_id = os.path.basename(label_path).replace(".labels.tsv", "")
        txt_path = os.path.join(folder_path, f"{article_id}.txt")

        if not os.path.exists(txt_path):
            print(f"❌ Missing article text for {article_id}")
            continue

        text = load_article(txt_path)
        labels = load_labels(label_path)
        fragments = get_matched_fragments(text, labels)
        all_fragments.extend(fragments)

    return pd.DataFrame(all_fragments)

if __name__ == "__main__":
    train_folder = "train"
    print(f"\n📂 Loading all training data from: {train_folder}")
    df = load_all_articles(train_folder)

    print(f"\n✅ Loaded {len(df)} labeled fragments from {df['article_id'].nunique()} articles.\n")
    print(df.head())

    # 💾 Save the data for ML training
    df.to_csv("labeled_fragments.csv", index=False)
    print("📁 Saved labeled_fragments.csv!")
