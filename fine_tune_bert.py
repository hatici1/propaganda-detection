import os
import pandas as pd
from transformers import BertTokenizerFast, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch

# ✅ Load train/dev from official folders (span-level classification)
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
                        aid, label, start, end = parts
                        start, end = int(start), int(end)
                        text = full_text[start:end]
                        data.append({
                            "article_id": article_id,
                            "technique": label,
                            "start": start,
                            "end": end,
                            "text": text,
                            "full_text": full_text
                        })
    return pd.DataFrame(data)

print("📂 Loading training and development data from official folders...")
train_df = load_data_from_folder("train")
dev_df = load_data_from_folder("dev")
print(f"✅ Loaded {len(train_df)} training and {len(dev_df)} development samples.")

# 🔢 Encode labels
label_list = sorted(set(train_df['technique'].unique()).union(set(dev_df['technique'].unique())))
label_to_id = {label: i for i, label in enumerate(label_list)}
id_to_label = {i: label for label, i in label_to_id.items()}

train_df['label_id'] = train_df['technique'].map(label_to_id)
dev_df['label_id'] = dev_df['technique'].map(label_to_id)

# 📦 Convert to Hugging Face datasets
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

def tokenize_function(example):
    return tokenizer(example["text"], truncation=True, padding='max_length', max_length=128)

train_dataset = Dataset.from_pandas(train_df[['text', 'label_id']].dropna()).map(tokenize_function, batched=True)
dev_dataset = Dataset.from_pandas(dev_df[['text', 'label_id']].dropna()).map(tokenize_function, batched=True)

train_dataset = train_dataset.rename_column("label_id", "labels")
dev_dataset = dev_dataset.rename_column("label_id", "labels")
train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
dev_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# 🤖 Load model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(label_list))

# ⚙️ Training arguments
training_args = TrainingArguments(
    output_dir="./bert_prop_model",
    evaluation_strategy="epoch",
    logging_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=8,  # 🔁 More epochs
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

# 🚀 Train
trainer.train()

# 📊 Evaluate
eval_results = trainer.evaluate()
print("\n📈 Evaluation Results:")
print(eval_results)

# 💾 Save model and tokenizer
model.save_pretrained("bert_prop_model")
tokenizer.save_pretrained("bert_prop_model")
print("✅ Fine-tuned BERT model saved to 'bert_prop_model'")
