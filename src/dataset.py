import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken

class GPTDataset(Dataset):
    def __init__(self, text, max_length=512, stride=512):
        self.input_ids = []
        self.target_ids = []
        tokenizer = tiktoken.get_encoding("gpt2")
        token_ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
        for i in range(0, len(token_ids)-max_length, stride):
            inp = token_ids[i:i+max_length]
            tgt = token_ids[i+1:i+max_length+1]
            self.input_ids.append(torch.tensor(inp, dtype=torch.long))
            self.target_ids.append(torch.tensor(tgt, dtype=torch.long))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

def create_loaders(train_txt, val_txt, test_txt, batch_size=4, max_length=512, stride=512):
    train_ds = GPTDataset(train_txt, max_length=max_length, stride=stride)
    val_ds = GPTDataset(val_txt, max_length=max_length, stride=stride)
    test_ds = GPTDataset(test_txt, max_length=max_length, stride=stride)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=2)
    return train_loader, val_loader, test_loader
