import os
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
                "technique": row.technique,
                "start": row.start_char,
                "end": row.end_char,
                "text": span
            })
        except Exception as e:
            print(f"Error extracting span: {e}")
    return fragments

def preview_sample(article_txt_path, label_tsv_path):
    print(f"\n📰 Loading article: {os.path.basename(article_txt_path)}")
    text = load_article(article_txt_path)
    labels = load_labels(label_tsv_path)
    print(f"Total labeled fragments: {len(labels)}")

    matched = get_matched_fragments(text, labels)
    for frag in matched[:5]:  # preview first 5
        print(f"\n[{frag['technique']}] ({frag['start']}–{frag['end']}):")
        print(f"👉 {frag['text']}\n{'-'*40}")

if __name__ == "__main__":
    # ✅ Preview one article from the train set
    article_id = "train/article111111113"
    txt_path = f"{article_id}.txt"
    label_path = f"{article_id}.labels.tsv"

    if os.path.exists(txt_path) and os.path.exists(label_path):
        preview_sample(txt_path, label_path)
    else:
        print("❌ File not found. Double-check the path.")
