# ✅ fine_tune_bert.py (improved version)
import os
import pandas as pd
from transformers import BertTokenizerFast, Trainer, TrainingArguments, BertConfig
from datasets import Dataset
import torch
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from my_model2 import CustomBertForTokenClassification

# ✅ Load train/dev from official folders
def load_data_from_folder(folder_path):
    data = []
    for fname in os.listdir(folder_path):
        if fname.endswith(".labels.tsv"):
            article_id = fname.replace(".labels.tsv", "")
            txt_file = os.path.join(folder_path, article_id + ".txt")
            label_file = os.path.join(folder_path, fname)

            if not os.path.exists(txt_file):
                continue

            with open(txt_file, encoding="utf-8") as f:
                full_text = f.read()

            with open(label_file, encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) == 4:
                        _, label, start, end = parts
                        start, end = int(start), int(end)
                        data.append({
                            "full_text": full_text,
                            "label": label,
                            "start": start,
                            "end": end
                        })
    return pd.DataFrame(data)

print("\U0001F4C2 Loading training and development data from official folders...")
train_df = load_data_from_folder("train")
dev_df = load_data_from_folder("dev")
print(f"✅ Loaded {len(train_df)} training and {len(dev_df)} development samples.")

# 🔢 Labels
label_list = sorted(set(train_df["label"]).union(dev_df["label"]))
label_list = ["O"] + label_list
label_to_id = {label: i for i, label in enumerate(label_list)}
id_to_label = {i: label for label, i in label_to_id.items()}

# 📦 Tokenizer
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

def encode_articles(df):
    grouped = df.groupby("full_text")
    features = []

    for full_text, group in grouped:
        encoding = tokenizer(full_text, return_offsets_mapping=True, truncation=True, padding="max_length", max_length=512)
        offsets = encoding["offset_mapping"]
        labels = [0] * len(offsets)

        for _, row in group.iterrows():
            for idx, (start, end) in enumerate(offsets):
                if start < row.end and end > row.start:
                    labels[idx] = label_to_id.get(row.label, 0)

        # 🔎 Debug how many non-O labels per article
        print(f"\u2192 {sum(1 for l in labels if l != 0)} tokens labeled in article")

        encoding["labels"] = labels
        encoding.pop("offset_mapping")
        features.append(encoding)

    return Dataset.from_list(features)

print("\U0001F501 Tokenizing and aligning labels...")
train_dataset = encode_articles(train_df)
dev_dataset = encode_articles(dev_df)

# ⚖️ Class weights
all_labels = []
for labels in train_dataset["labels"]:
    all_labels.extend([l for l in labels if l != 0])  # Exclude 'O' from weight calc

existing_classes = np.unique(all_labels)
computed_weights = compute_class_weight(class_weight='balanced', classes=existing_classes, y=all_labels)
class_weights = np.ones(len(label_list), dtype=np.float32)
for i, cls in enumerate(existing_classes):
    class_weights[cls] = computed_weights[i]
class_weights = torch.tensor(class_weights, dtype=torch.float)

# 🤖 Load model
config = BertConfig.from_pretrained("bert-base-uncased", num_labels=len(label_list))
model = CustomBertForTokenClassification(config, class_weights=class_weights)

# ⚙️ Training args
training_args = TrainingArguments(
    output_dir="./bert_prop_model",
    evaluation_strategy="epoch",
    logging_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=12,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    logging_dir="./logs"
)

# 🏋️ Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    tokenizer=tokenizer,
)

trainer.train()

# 📊 Evaluate
eval_results = trainer.evaluate()
print("\n\U0001F4C8 Evaluation Results:", eval_results)

# 💾 Save
model.save_pretrained("bert_prop_model")
tokenizer.save_pretrained("bert_prop_model")
print("\u2705 Model and tokenizer saved to 'bert_prop_model'")
