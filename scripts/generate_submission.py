import os
import torch
from transformers import BertTokenizerFast, BertForTokenClassification
from datasets import Dataset
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Load fine-tuned model and tokenizer
model = BertForTokenClassification.from_pretrained("bert_prop_model").to(device)
tokenizer = BertTokenizerFast.from_pretrained("bert_prop_model")

# ✅ Load label mapping
# Custom label mapping (should match your training)
label_list = [
    'Appeal_to_Authority', 'Appeal_to_fear-prejudice', 'Bandwagon',
    'Black-and-White_Fallacy', 'Causal_Oversimplification', 'Doubt',
    'Exaggeration,Minimisation', 'Flag-Waving', 'Loaded_Language',
    'Name_Calling,Labeling', 'Obfuscation,Intentional_Vagueness,Confusion',
    'Red_Herring', 'Reductio_ad_hitlerum', 'Repetition', 'Slogans',
    'Straw_Men', 'Thought-terminating_Cliches', 'Whataboutism'
]
id_to_label = {i: label for i, label in enumerate(label_list)}

# ✅ Directory with test articles
test_dir = "test"  # or "dev"
article_files = [f for f in os.listdir(test_dir) if f.endswith(".txt")]

submissions = []

print(f"🔍 Generating predictions for {len(article_files)} articles...")

for filename in article_files:
    article_id = filename.replace(".txt", "").replace("article", "")
    with open(os.path.join(test_dir, filename), "r", encoding="utf-8") as f:
        text = f.read()

    # Tokenize with offset mappings
    tokens = tokenizer(text, return_offsets_mapping=True, return_tensors="pt", truncation=True, padding=True, max_length=512)
    tokens = {k: v.to(device) for k, v in tokens.items()}
    offset_mapping = tokens.pop("offset_mapping")[0]

    with torch.no_grad():
        outputs = model(**tokens)
        predictions = torch.argmax(outputs.logits, dim=-1)[0].cpu().numpy()

    # Decode spans
    previous_label = None
    span_start = None

    for idx, label_id in enumerate(predictions):
        label = id_to_label[label_id]
        if label == 'O':
            if previous_label and previous_label != 'O' and span_start is not None:
                start_char = offset_mapping[span_start][0].item()
                end_char = offset_mapping[idx - 1][1].item()
                if end_char > start_char:
                    submissions.append([article_id, label, start_char, end_char])

                span_start = None
            previous_label = None
        else:
            if label != previous_label:
                if previous_label and span_start is not None:
                    start_char = offset_mapping[span_start][0].item()
                    end_char = offset_mapping[idx - 1][1].item()
                    if end_char > start_char:
                        submissions.append([article_id, label, start_char, end_char])

                span_start = idx
            previous_label = label

    # Handle case if last token is a label
    # Handle case if last token is a label
        if previous_label and span_start is not None:
            start_char = offset_mapping[span_start][0].item()
            end_char = offset_mapping[len(predictions) - 1][1].item()
            if end_char > start_char:
                submissions.append([article_id, label, start_char, end_char])


# 🔽 Save submission file
submission_df = pd.DataFrame(submissions, columns=["article_id", "technique", "start_char", "end_char"])
submission_df.to_csv("submission.tsv", sep="\t", index=False, header=False)
print(f"\n✅ Saved prediction results to 'submission.tsv' ({len(submission_df)} rows)")
