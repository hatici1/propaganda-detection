from transformers import BertTokenizerFast, BertForTokenClassification
import torch

model = BertForTokenClassification.from_pretrained("bert_prop_model")
tokenizer = BertTokenizerFast.from_pretrained("bert_prop_model")
model.eval()

sample_text = "The government is trying to deceive the people and confuse the facts."
inputs = tokenizer(sample_text, return_tensors="pt", return_offsets_mapping=True, truncation=True)
with torch.no_grad():
    outputs = model(**{k: v for k, v in inputs.items() if k != "offset_mapping"})
preds = torch.argmax(outputs.logits, dim=-1)[0].tolist()

id_to_label = {i: label for i, label in enumerate([
    'O', 'Appeal_to_Authority', 'Appeal_to_fear-prejudice', 'Bandwagon',
    'Black-and-White_Fallacy', 'Causal_Oversimplification', 'Doubt',
    'Exaggeration,Minimisation', 'Flag-Waving', 'Loaded_Language',
    'Name_Calling,Labeling', 'Obfuscation,Intentional_Vagueness,Confusion',
    'Red_Herring', 'Reductio_ad_hitlerum', 'Repetition', 'Slogans',
    'Straw_Men', 'Thought-terminating_Cliches', 'Whataboutism'
])}

print([id_to_label[p] for p in preds])
