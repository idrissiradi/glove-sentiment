import numpy as np
import torch

from model import SentimentLSTM

MAX_LEN = 64


def tokenize(sentence):
    return sentence.lower().strip().split()


def encode(sentence, vocab, max_len):
    tokens = tokenize(sentence)
    ids = [vocab.get(t, 1) for t in tokens]
    if len(ids) < max_len:
        ids += [0] * (max_len - len(ids))
    else:
        ids = ids[:max_len]
    return ids


def predict(sentences, model_path="data/best_model_finetuned.pt"):
    vocab = np.load("data/vocab.npy", allow_pickle=True).item()
    embedding_matrix = np.load("data/embedding_matrix.npy")

    model = SentimentLSTM(embedding_matrix, freeze=False)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    print(f"\n{'=' * 50}")
    print(f"{'Sentence':<40} {'Score':>6}  Prediction")
    print(f"{'=' * 50}")

    with torch.no_grad():
        for sentence in sentences:
            ids = encode(sentence, vocab, MAX_LEN)
            x = torch.tensor([ids], dtype=torch.long)
            logit = model(x)
            prob = torch.sigmoid(logit).item()
            label = "positive ✓" if prob >= 0.5 else "negative ✗"
            print(f"{sentence:<40} {prob:>6.3f}  {label}")


if __name__ == "__main__":
    test_sentences = [
        # clearly positive
        "this movie was absolutely wonderful",
        "a charming and affecting journey",
        "the best film i have seen this year",
        # clearly negative
        "completely boring and a waste of time",
        "terrible acting and a awful script",
        "i hated every minute of this film",
        # tricky
        "not bad at all",
        "could have been much better",
        "not the worst film ever made",
    ]
    predict(test_sentences)
