import pandas as pd
import torch
from torch.nn import CrossEntropyLoss
from transformers import (AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments,
                          DataCollatorForTokenClassification)
from sklearn.model_selection import train_test_split
from dataset import NERDataset

# Read ready dataset
df = pd.read_pickle("dataset/ready_dataset.pkl")

# Data split
train_data, eval_data = train_test_split(df, test_size=0.2, random_state=42)

# Import tokenizer
tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")

# Create labels
id2label = {
    0: "O",
    1: "I-LOC",
    2: "B-LOC"
}
label2id = {
    "O": 0,
    "I-LOC": 1,
    "B-LOC": 2
}

# Dataset wrapper
train_data = NERDataset(train_data, tokenizer, label2id)
print(train_data[0])
eval_data = NERDataset(eval_data, tokenizer, label2id)


# Import model
model = AutoModelForTokenClassification.from_pretrained(
    "dslim/bert-base-NER", num_labels=len(id2label), id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True
)

# Data collator
data_collator = DataCollatorForTokenClassification(tokenizer)


# Trainer
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        logits = outputs.logits

        labels = inputs['labels']

        loss_fct = CrossEntropyLoss(weight=torch.tensor([0.1, 0.8, 0.8]))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))

        return (loss, outputs) if return_outputs else loss


training_args = TrainingArguments(
    output_dir="model",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=4,
    num_train_epochs=1,
    logging_dir='./logs',
)

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=eval_data,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()
