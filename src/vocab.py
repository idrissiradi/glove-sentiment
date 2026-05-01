# vocab.py
import numpy as np
from datasets import load_dataset

# ── config ──────────────────────────────────────────
GLOVE_PATH = "data/glove.6B.100d.txt"
EMBED_DIM = 100
MAX_VOCAB = 20000  # keep only top N most frequent words
MAX_LEN = 64  # pad/truncate sentences to this length


# ── simple tokenizer ─────────────────────────────────
def tokenize(sentence):
    return sentence.lower().strip().split()


# ── build vocabulary from training set ───────────────
def build_vocab(dataset, max_vocab):
    from collections import Counter

    counter = Counter()
    for example in dataset["train"]:
        tokens = tokenize(example["sentence"])
        counter.update(tokens)

    # special tokens
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, _ in counter.most_common(max_vocab - 2):
        vocab[word] = len(vocab)
    return vocab


# ── load GloVe into a dict ────────────────────────────
def load_glove(glove_path):
    glove = {}
    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            word = parts[0]
            vec = np.array(parts[1:], dtype=np.float32)
            glove[word] = vec
    print(f"GloVe loaded: {len(glove)} words")
    return glove


# ── build embedding matrix E ──────────────────────────
def build_embedding_matrix(vocab, glove, embed_dim):
    matrix = np.zeros((len(vocab), embed_dim), dtype=np.float32)

    # use mean GloVe vector for unknown words
    glove_vectors = np.stack(list(glove.values()))
    unk_vec = glove_vectors.mean(axis=0)

    found, not_found = 0, 0
    for word, idx in vocab.items():
        if word in glove:
            matrix[idx] = glove[word]
            found += 1
        else:
            matrix[idx] = unk_vec
            not_found += 1

    print(f"Embedding matrix: {matrix.shape}")
    print(f"Words found in GloVe: {found} | not found: {not_found}")
    return matrix


# ── run ───────────────────────────────────────────────
if __name__ == "__main__":
    dataset = load_dataset("stanfordnlp/sst2")
    vocab = build_vocab(dataset, MAX_VOCAB)
    glove = load_glove(GLOVE_PATH)
    matrix = build_embedding_matrix(vocab, glove, EMBED_DIM)

    # quick sanity check — similar words should have similar vectors
    def cosine(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    good_idx = vocab.get("good")
    great_idx = vocab.get("great")
    terrible_idx = vocab.get("terrible")

    print(
        f"\ngood vs great:     {cosine(matrix[good_idx], matrix[great_idx]):.4f}  (expect HIGH)"
    )
    print(
        f"good vs terrible:  {cosine(matrix[good_idx], matrix[terrible_idx]):.4f}  (expect LOW)"
    )

    # save for later
    np.save("data/embedding_matrix.npy", matrix)
    np.save("data/vocab.npy", vocab)
    print("\nSaved embedding_matrix.npy and vocab.npy")
