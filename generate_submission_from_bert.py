import os
import torch
from transformers import BertTokenizerFast, BertForSequenceClassification
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load fine-tuned BERT model and tokenizer
model = BertForSequenceClassification.from_pretrained("bert_prop_model").to(device)
tokenizer = BertTokenizerFast.from_pretrained("bert_prop_model")

# Labels (must match your training set!)
label_list = [
    'Appeal_to_Authority', 'Appeal_to_fear-prejudice', 'Bandwagon',
    'Black-and-White_Fallacy', 'Causal_Oversimplification', 'Doubt',
    'Exaggeration,Minimisation', 'Flag-Waving', 'Loaded_Language',
    'Name_Calling,Labeling', 'Obfuscation,Intentional_Vagueness,Confusion',
    'Red_Herring', 'Reductio_ad_hitlerum', 'Repetition', 'Slogans',
    'Straw_Men', 'Thought-terminating_Cliches', 'Whataboutism'
]
id_to_label = {i: label for i, label in enumerate(label_list)}

# Directory containing test articles
test_dir = "test"
submissions = []

print("🔍 Predicting propaganda techniques in test fragments...")

for fname in os.listdir(test_dir):
    if fname.endswith(".txt"):
        article_id = fname.replace(".txt", "").replace("article", "")
        with open(os.path.join(test_dir, fname), encoding="utf-8") as f:
            full_text = f.read()

        start = 0
        for sentence in full_text.split("."):
            sentence = sentence.strip()
            if not sentence:
                continue

            end = start + len(sentence)

            # Tokenize and predict
            encoded = tokenizer(sentence, return_tensors="pt", truncation=True, padding='max_length', max_length=128).to(device)
            with torch.no_grad():
                outputs = model(**encoded)
                pred_id = torch.argmax(outputs.logits, dim=1).item()
                pred_label = id_to_label[pred_id]

            # Only save if it's not a dummy prediction (optional)
            submissions.append([article_id, pred_label, start, end])
            start = end + 1  # move to next sentence

# Save submission
submission_df = pd.DataFrame(submissions, columns=["article_id", "technique", "start", "end"])
submission_df.to_csv("submission_from_bert.tsv", sep="\t", index=False, header=False)
print(f"✅ Saved BERT submission to 'submission_from_bert.tsv' ({len(submission_df)} rows)")
