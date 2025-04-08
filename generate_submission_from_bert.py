import os
import torch
import pandas as pd
from transformers import BertTokenizerFast, BertForTokenClassification

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Load fine-tuned model and tokenizer
model = BertForTokenClassification.from_pretrained("bert_prop_model").to(device)
tokenizer = BertTokenizerFast.from_pretrained("bert_prop_model")

# ✅ Label list (including "O")
label_list = [
    'O',  # must be included for token classification
    'Appeal_to_Authority', 'Appeal_to_fear-prejudice', 'Bandwagon',
    'Black-and-White_Fallacy', 'Causal_Oversimplification', 'Doubt',
    'Exaggeration,Minimisation', 'Flag-Waving', 'Loaded_Language',
    'Name_Calling,Labeling', 'Obfuscation,Intentional_Vagueness,Confusion',
    'Red_Herring', 'Reductio_ad_hitlerum', 'Repetition', 'Slogans',
    'Straw_Men', 'Thought-terminating_Cliches', 'Whataboutism'
]
id_to_label = {i: label for i, label in enumerate(label_list)}

# ✅ Directory with test articles
test_dir = "test"
submissions = []

print("🔍 Predicting propaganda techniques in test fragments...")

for fname in os.listdir(test_dir):
    if fname.endswith(".txt"):
        article_id = fname.replace(".txt", "").replace("article", "")
        with open(os.path.join(test_dir, fname), encoding="utf-8") as f:
            full_text = f.read()

        # Tokenize with offset mappings
        tokens = tokenizer(
            full_text,
            return_offsets_mapping=True,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )
        tokens = {k: v.to(device) for k, v in tokens.items()}
        offset_mapping = tokens["offset_mapping"][0]
        input_ids = tokens["input_ids"][0]
        attention_mask = tokens["attention_mask"][0]

        # Remove offset_mapping from model input
        del tokens["offset_mapping"]

        with torch.no_grad():
            outputs = model(**tokens)
            predicted_ids = torch.argmax(outputs.logits, dim=-1)[0].cpu().tolist()

        # Extract spans
        previous_label = 'O'
        span_start = None

        for idx, label_id in enumerate(predicted_ids):
            label = id_to_label[label_id]

            # Ignore padding
            if attention_mask[idx].item() == 0:
                continue

            if label != 'O':
                if label != previous_label:
                    if previous_label != 'O' and span_start is not None:
                        start_char = offset_mapping[span_start][0].item()
                        end_char = offset_mapping[idx - 1][1].item()
                        if end_char > start_char:
                            submissions.append([article_id, previous_label, start_char, end_char])
                    span_start = idx
                previous_label = label
            else:
                if previous_label != 'O' and span_start is not None:
                    start_char = offset_mapping[span_start][0].item()
                    end_char = offset_mapping[idx - 1][1].item()
                    if end_char > start_char:
                        submissions.append([article_id, previous_label, start_char, end_char])
                    span_start = None
                previous_label = 'O'

        # Handle last span
        if previous_label != 'O' and span_start is not None:
            start_char = offset_mapping[span_start][0].item()
            end_char = offset_mapping[len(predicted_ids) - 1][1].item()
            if end_char > start_char:
                submissions.append([article_id, previous_label, start_char, end_char])

# ✅ Save submission
submission_df = pd.DataFrame(submissions, columns=["article_id", "technique", "start_char", "end_char"])
submission_df.to_csv("submission_from_bert.tsv", sep="\t", index=False, header=False)
print(f"\n✅ Saved BERT submission to 'submission_from_bert.tsv' ({len(submission_df)} rows)")
