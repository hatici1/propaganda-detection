import os
import torch
import pandas as pd
from transformers import BertTokenizerFast, BertForTokenClassification

# ✅ Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Load fine-tuned model and tokenizer
model = BertForTokenClassification.from_pretrained("bert_prop_model").to(device)
tokenizer = BertTokenizerFast.from_pretrained("bert_prop_model")

# ✅ Label mapping
label_list = [
    'O',  # background class
    'Appeal_to_Authority', 'Appeal_to_fear-prejudice', 'Bandwagon',
    'Black-and-White_Fallacy', 'Causal_Oversimplification', 'Doubt',
    'Exaggeration,Minimisation', 'Flag-Waving', 'Loaded_Language',
    'Name_Calling,Labeling', 'Obfuscation,Intentional_Vagueness,Confusion',
    'Red_Herring', 'Reductio_ad_hitlerum', 'Repetition', 'Slogans',
    'Straw_Men', 'Thought-terminating_Cliches', 'Whataboutism'
]
id_to_label = {i: label for i, label in enumerate(label_list)}

# ✅ Predict and extract spans
def predict_on_test_articles(test_dir="test"):
    submissions = []
    print("\U0001F50D Predicting propaganda techniques in test articles...")

    for fname in sorted(os.listdir(test_dir)):
        if not fname.endswith(".txt"):
            continue

        article_id = fname.replace(".txt", "")
        with open(os.path.join(test_dir, fname), encoding="utf-8") as f:
            full_text = f.read()

        encoded = tokenizer(
            full_text,
            return_offsets_mapping=True,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt"
        )

        offset_mapping = encoded.pop("offset_mapping")[0]
        input_ids = encoded["input_ids"][0]
        attention_mask = encoded["attention_mask"][0]

        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.no_grad():
            outputs = model(**encoded)
            preds = torch.argmax(outputs.logits, dim=-1)[0].cpu().tolist()

        current_label = "O"
        span_start = None

        for i, pred_id in enumerate(preds):
            if attention_mask[i] == 0:
                continue  # skip padding

            label = id_to_label[pred_id]

            if label != "O":
                if label != current_label:
                    if current_label != "O" and span_start is not None:
                        start = offset_mapping[span_start][0].item()
                        end = offset_mapping[i - 1][1].item()
                        if end > start:
                            submissions.append([article_id, current_label, start, end])
                    span_start = i
                current_label = label
            else:
                if current_label != "O" and span_start is not None:
                    start = offset_mapping[span_start][0].item()
                    end = offset_mapping[i - 1][1].item()
                    if end > start:
                        submissions.append([article_id, current_label, start, end])
                    span_start = None
                current_label = "O"

        # Final span
        if current_label != "O" and span_start is not None:
            start = offset_mapping[span_start][0].item()
            end = offset_mapping[len(preds) - 1][1].item()
            if end > start:
                submissions.append([article_id, current_label, start, end])

    return submissions

# ✅ Run and save
submissions = predict_on_test_articles("test")
submission_df = pd.DataFrame(submissions, columns=["article_id", "technique", "start", "end"])
submission_df.to_csv("submission_from_bert.tsv", sep="\t", index=False, header=False)
print(f"\n✅ Saved BERT submission to 'submission_from_bert.tsv' ({len(submission_df)} rows)")
