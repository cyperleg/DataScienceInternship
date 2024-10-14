import pandas as pd
import torch
from nervaluate import Evaluator
from transformers import AutoModelForTokenClassification, AutoTokenizer

# ChatGPT inference dataset
dataset = [
    ("Mount Everest is the highest mountain in the world.", ["B-LOC", "I-LOC", "O", "O", "O", "O", "O", "O", "O"]),
    ("Kilimanjaro is a dormant volcano in Tanzania.", ["B-LOC", "O", "O", "O", "O", "O", "B-LOC", "O"]),
    ("The Andes Mountains stretch along South America's western side.", ["O", "B-LOC", "I-LOC", "O", "O", "O", "B-LOC", "I-LOC", "O", "O"]),
    ("Mount Fuji is Japan's tallest peak.", ["B-LOC", "I-LOC", "O", "B-LOC", "O", "O", "O"]),
    ("The Rocky Mountains are located in North America.", ["O", "B-LOC", "I-LOC", "O", "O", "O", "B-LOC", "I-LOC", "O"]),
    ("The Alps are a popular destination for skiing.", ["O", "B-LOC", "O", "O", "O", "O", "O", "O"]),
    ("Mont Blanc is the highest peak in the Alps.", ["B-LOC", "I-LOC", "O", "O", "O", "O", "O", "B-LOC", "O"]),
    ("Mount Elbrus is the tallest mountain in Europe.", ["B-LOC", "I-LOC", "O", "O", "O", "O", "O", "O"]),
    ("Denali, located in Alaska, is North America's highest peak.", ["B-LOC", "O", "O", "O", "B-LOC", "O", "O", "B-LOC", "I-LOC", "O", "O", "O"]),
    ("Mount Kosciuszko is the highest mountain in Australia.", ["B-LOC", "I-LOC", "O", "O", "O", "O", "O", "B-LOC", "O"]),
    ("The Appalachian Mountains are located in eastern North America.", ["O", "B-LOC", "I-LOC", "O", "O", "O", "O", "B-LOC", "I-LOC", "O"]),
    ("Mount Rainier is a massive stratovolcano located in Washington.", ["B-LOC", "I-LOC", "O", "O", "O", "O", "O", "B-LOC", "O"]),
    ("The Ural Mountains form part of the boundary between Europe and Asia.", ["O", "B-LOC", "I-LOC", "O", "O", "O", "O", "B-LOC", "O", "O", "B-LOC", "O", "O"]),
    ("The Pyrenees form the natural border between France and Spain.", ["O", "B-LOC", "O", "O", "O", "O", "O", "B-LOC", "O", "O", "B-LOC", "O"]),
    ("Mount Ararat is a dormant volcano in Turkey.", ["B-LOC", "I-LOC", "O", "O", "O", "O", "O", "B-LOC", "O"]),
    ("Aconcagua is the highest peak in South America.", ["B-LOC", "O", "O", "O", "O", "O", "B-LOC", "I-LOC", "O"]),
    ("Mount Kenya is the second-highest peak in Africa.", ["B-LOC", "I-LOC", "O", "O", "O", "O", "O", "B-LOC", "O"]),
    ("The Caucasus Mountains stretch between the Black Sea and the Caspian Sea.", ["O", "B-LOC", "I-LOC", "O", "O", "O", "B-LOC", "I-LOC", "O", "O", "B-LOC", "I-LOC", "O"]),
    ("Mount Etna is one of the most active volcanoes in the world.", ["B-LOC", "I-LOC", "O", "O", "O", "O", "O", "O", "O", "O", "O"]),
    ("The Carpathian Mountains are located in Eastern Europe.", ["O", "B-LOC", "I-LOC", "O", "O", "O", "B-LOC", "I-LOC", "O"])
]

# Label map
label_map = {0: "O", 1: "B-LOC", 2: "I-LOC"}

# Creating dataset
df = pd.DataFrame(dataset, columns=["sentence", "labels"])
df['labels'].apply(lambda x: x.insert(0, 'O'))

# Import model
model = AutoModelForTokenClassification.from_pretrained("model/checkpoint-46")
tokenizer = AutoTokenizer.from_pretrained("model/checkpoint-46")
inputs = tokenizer(df['sentence'].tolist(), padding=True, truncation=True, return_tensors="pt")

# Getting predictions
with torch.no_grad():
    outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
logits = outputs.logits
predicted_token_ids = torch.argmax(logits, dim=2)
predicted_labels = [[label_map[x.item()] for x in token_id] for token_id in predicted_token_ids]

# Adding padding dataset
max_len = len(predicted_labels[0])
df['labels'] = df['labels'].apply(lambda x: x + ['O'] * (max_len - len(x)))

tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

# TODO Troubles with evalutator
evaluator = Evaluator(df['labels'].tolist(), predicted_labels, tags=['O', 'B-LOC', 'I-LOC'], loader="list")

print(evaluator.evaluate()[1])
