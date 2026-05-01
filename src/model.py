import numpy as np
import torch
import torch.nn as nn


class SentimentLSTM(nn.Module):
    def __init__(
        self, embedding_matrix, hidden_dim=128, num_layers=2, dropout=0.3, freeze=True
    ):
        super(SentimentLSTM, self).__init__()
        vocab_size, embed_dim = embedding_matrix.shape

        # ── Embedding layer ──────────────────────────────────────
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.embedding.weight = nn.Parameter(
            torch.tensor(embedding_matrix, dtype=torch.float32),
            requires_grad=not freeze,
        )

        # ── LSTM ─────────────────────────────────────────────────
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        # ✅ FIX: Proper LSTM initialization
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                param.data.fill_(0)
                n = param.size(0)
                param.data[n // 4 : n // 2].fill_(1.0)  # forget gate = 1

        # ── Classifier head ──────────────────────────────────────
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        last_hidden = self.dropout(hidden[-1])

        out = self.fc(last_hidden)
        # out = self.sigmoid(out)

        return out.squeeze()


if __name__ == "__main__":
    # sanity check — does the model run without errors?
    embedding_matrix = np.load("data/embedding_matrix.npy")
    model = SentimentLSTM(embedding_matrix, freeze=True)

    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(
        f"Trainable parameters (frozen E): {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )

    # fake batch — 4 sentences of length 64
    fake_input = torch.randint(0, 1000, (4, 64))
    output = model(fake_input)
    print(f"\nOutput shape: {output.shape}")
    print(f"Output values: {output}")
