from calendar import c

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataset import SSTDataset
from src.model import SentimentLSTM

# ── config ────────────────────────────────────────────────────
EPOCHS = 5
BATCH_SIZE = 32
LR = 1e-3
HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── helper: compute accuracy ──────────────────────────────────
def accuracy(preds, labels):
    probs = torch.sigmoid(preds)
    binary_preds = (probs >= 0.5).float()
    correct = (binary_preds == labels).float().sum()
    return correct / len(labels)


# ── one full epoch of training ────────────────────────────────
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, total_acc = 0, 0

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()
        preds = model(x)
        loss = criterion(preds, y)
        loss.backward()

        # gradient clipping — prevents exploding gradients in LSTM
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()
        total_acc += accuracy(preds, y).item()

    avg_loss = total_loss / len(loader)
    avg_acc = total_acc / len(loader)
    return avg_loss, avg_acc


# ── one full epoch of evaluation ─────────────────────────────
def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss, total_acc = 0, 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            preds = model(x)
            loss = criterion(preds, y)

            total_loss += loss.item()
            total_acc += accuracy(preds, y).item()

    avg_loss = total_loss / len(loader)
    avg_acc = total_acc / len(loader)
    return avg_loss, avg_acc


# ── main training function ────────────────────────────────────
def train(freeze=True):
    print(f"\n{'=' * 50}")
    print(f"Phase: {'FROZEN embeddings' if freeze else 'FINE-TUNED embeddings'}")
    print(f"{'=' * 50}")

    # load data
    vocab = np.load("data/vocab.npy", allow_pickle=True).item()
    train_ds = SSTDataset("train", vocab)
    val_ds = SSTDataset("validation", vocab)
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # load model
    embedding_matrix = np.load("data/embedding_matrix.npy")
    model = SentimentLSTM(
        embedding_matrix,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        freeze=freeze,
    ).to(DEVICE)

    # loss function — binary cross entropy
    # criterion = nn.BCELoss()
    criterion = nn.BCEWithLogitsLoss()

    # optimizer — Adam
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=1e-5
    )

    best_val_acc = 0
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc = eval_epoch(model, val_loader, criterion)

        print(
            f"Epoch {epoch}/{EPOCHS} "
            f"| train loss {train_loss:.4f} acc {train_acc:.4f} "
            f"| val loss {val_loss:.4f} acc {val_acc:.4f}"
        )

        # save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                model.state_dict(),
                f"data/best_model_{'frozen' if freeze else 'finetuned'}.pt",
            )
            print(f"  ✓ best model saved (val acc {best_val_acc:.4f})")

    print(f"\nBest validation accuracy: {best_val_acc:.4f}")
    return best_val_acc


# ── run both phases and compare ───────────────────────────────
if __name__ == "__main__":
    print(f"Device: {DEVICE}")

    # Phase 1 — frozen embeddings (transfer learning)
    frozen_acc = train(freeze=True)

    # Phase 2 — fine-tuned embeddings
    finetuned_acc = train(freeze=False)

    # comparison — the whole point of this project
    print(f"\n{'=' * 50}")
    print(f"RESULTS COMPARISON")
    print(f"{'=' * 50}")
    print(f"Frozen embeddings:    {frozen_acc:.4f}")
    print(f"Fine-tuned embeddings:{finetuned_acc:.4f}")
    print(f"Difference:           {finetuned_acc - frozen_acc:+.4f}")
