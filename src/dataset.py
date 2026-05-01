# dataset.py
import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset

MAX_LEN = 64


def tokenize(sentence):
    return sentence.lower().strip().split()


def encode(sentence, vocab, max_len):
    tokens = tokenize(sentence)
    ids = [vocab.get(t, 1) for t in tokens]  # 1 = <UNK>
    # pad or truncate to max_len
    if len(ids) < max_len:
        ids += [0] * (max_len - len(ids))  # 0 = <PAD>
    else:
        ids = ids[:max_len]
    return ids


class SSTDataset(Dataset):
    def __init__(self, split, vocab):
        dataset = load_dataset("stanfordnlp/sst2")
        self.data = dataset[split]
        self.vocab = vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        ids = encode(example["sentence"], self.vocab, MAX_LEN)
        label = example["label"]
        return (
            torch.tensor(ids, dtype=torch.long),
            torch.tensor(label, dtype=torch.float),
        )


if __name__ == "__main__":
    vocab = np.load("data/vocab.npy", allow_pickle=True).item()

    train_ds = SSTDataset("train", vocab)
    val_ds = SSTDataset("validation", vocab)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4)

    # sanity check
    x, y = next(iter(train_loader))
    print(f"Batch input shape:  {x.shape}")  # (32, 64)
    print(f"Batch label shape:  {y.shape}")  # (32,)
    print(f"First sequence:     {x[0]}")
    print(f"First label:        {y[0]}")
