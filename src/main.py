import numpy as np
import torch


def main():
    print("Hello from glove-sentiment!")
    print("=== CHECK 1: GloVe File ===")
    try:
        with open("data/glove.6B.100d.txt", "r") as f:
            first_line = f.readline()
            print(f"First line: {first_line[:50]}...")
            print(f"✅ GloVe file exists")
    except FileNotFoundError:
        print("❌ GloVe file NOT FOUND at data/glove.6B.100d.txt")
        print("   Download from: https://nlp.stanford.edu/projects/glove/")

    print("\n=== CHECK 2: Embedding Matrix ===")
    matrix = np.load("data/embedding_matrix.npy")
    print(f"Shape: {matrix.shape}")
    print(f"Mean: {matrix.mean():.4f}, Std: {matrix.std():.4f}")
    if matrix.std() < 0.01:
        print("❌ Matrix has no variance — GloVe didn't load properly!")
    else:
        print("✅ Matrix looks healthy")

    print("\n=== CHECK 3: Vocab ===")
    vocab = np.load("data/vocab.npy", allow_pickle=True).item()
    print(f"Vocab size: {len(vocab)}")
    print(f"'the' → {vocab.get('the')}")
    print(f"'good' → {vocab.get('good')}")
    if vocab.get("the") is None:
        print("❌ Common words missing from vocab!")
    else:
        idx = vocab["the"]
        print(f"Matrix row for 'the': {matrix[idx][:5]}")

    print("\n=== CHECK 4: Model Output ===")
    from model import SentimentLSTM

    model = SentimentLSTM(matrix, freeze=True)
    fake_input = torch.randint(0, 100, (4, 64))
    output = model(fake_input)
    print(f"Output: {output}")
    print(f"Output mean: {output.mean():.4f}")
    if abs(output.mean() - 0.5) < 0.01 and output.std() < 0.01:
        print("❌ Output is constant — model broken!")
    else:
        print("✅ Output varies — model can learn")


if __name__ == "__main__":
    main()
