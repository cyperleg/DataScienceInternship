import torch
from torch.utils.data import Dataset


# Dataset class for batching
class NERDataset(Dataset):
    def __init__(self, df, tokenizer, mapping, max_length=200):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mapping = mapping

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Tokenizing
        encoding = self.tokenizer(
            self.df.text[idx],
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt',
            return_attention_mask=True,
            return_token_type_ids=True
        )

        # Labels creating
        labels = [self.mapping[label] for label in self.df.markup[idx]]
        labels += [-100] * (self.max_length - len(labels))  # Предполагаем, что O имеет значение 0
        labels.insert(0, -100)
        labels = labels[:self.max_length]

        # Debugging
        # print(encoding['input_ids'])
        # print(self.tokenizer.decode(encoding['input_ids'][0]))
        # print(labels)

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(labels)
        }