import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import BertTokenizerFast, BertForTokenClassification, Trainer, TrainingArguments
from transformers import DataCollatorForTokenClassification
import torch
from datasets import Dataset

# ✅ Load cleaned dataset
print("📂 Loading cleaned labeled fragments...")
df = pd.read_csv("labeled_fragments_cleaned.csv")

# 🔢 Encode labels
label_list = sorted(df['technique'].unique())
label_to_id = {label: i for i, label in enumerate(label_list)}
id_to_label = {i: label for label, i in label_to_id.items()}
df['label_id'] = df['technique'].map(label_to_id)

# 🧠 Load tokenizer
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

# 📦 Prepare dataset in HuggingFace format
dataset = Dataset.from_pandas(df[['text', 'label_id']].dropna())

# Tokenize
encoded = dataset.map(lambda x: tokenizer(x['text'], truncation=True, padding='max_length', max_length=128), batched=True)
encoded = encoded.rename_column("label_id", "labels")
encoded.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# 🔀 Split
train_test = encoded.train_test_split(test_size=0.2, seed=42)

# 🤖 Model
model = BertForTokenClassification.from_pretrained("bert-base-uncased", num_labels=len(label_list))

# Training arguments
training_args = TrainingArguments(
    output_dir="./bert_prop_model",
    evaluation_strategy="epoch",
    logging_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=4,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
)

# Data collator
data_collator = DataCollatorForTokenClassification(tokenizer)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_test["train"],
    eval_dataset=train_test["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# 🚀 Train
trainer.train()

# 📊 Evaluate
eval_results = trainer.evaluate()
print("\n📈 Evaluation Results:")
print(eval_results)

# Save model
model.save_pretrained("bert_prop_model")
tokenizer.save_pretrained("bert_prop_model")
print("✅ Model and tokenizer saved to 'bert_prop_model'")
