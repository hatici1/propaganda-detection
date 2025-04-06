import pandas as pd
import torch
from transformers import BertTokenizerFast, BertForTokenClassification
from sklearn.metrics import classification_report
from datasets import Dataset

# Load model and tokenizer
model = BertForTokenClassification.from_pretrained("bert_prop_model")
tokenizer = BertTokenizerFast.from_pretrained("bert_prop_model")

# Load and prepare data
df = pd.read_csv("labeled_fragments_cleaned.csv").dropna(subset=["text", "technique"])
label_list = sorted(df["technique"].unique())
label_to_id = {label: i for i, label in enumerate(label_list)}
id_to_label = {i: label for label, i in label_to_id.items()}
df["label_id"] = df["technique"].map(label_to_id)

dataset = Dataset.from_pandas(df[["text", "label_id"]])
encoded = dataset.map(lambda x: tokenizer(x["text"], truncation=True, padding="max_length", max_length=128), batched=True)
encoded = encoded.rename_column("label_id", "labels")
encoded.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# Hugging Face split
split_dataset = encoded.train_test_split(test_size=0.2, seed=42)
test_set = split_dataset["test"]

# Predict CLS token label
model.eval()
preds, true_labels = [], []

for batch in torch.utils.data.DataLoader(test_set, batch_size=16):
    with torch.no_grad():
        outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        logits = outputs.logits  # shape: (batch_size, seq_len, num_labels)
        cls_logits = logits[:, 0, :]  # take the logits for [CLS] token
        pred_ids = torch.argmax(cls_logits, axis=1).cpu().numpy()
        label_ids = batch["labels"].cpu().numpy()

        preds.extend(pred_ids)
        true_labels.extend(label_ids)

# Print results
print("\n📊 Classification Report (BERT using [CLS] token):")
print(classification_report(true_labels, preds, target_names=label_list))
