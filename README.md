# Sentiment Analysis with GloVe + LSTM

Binary sentiment classifier on SST-2 using pre-trained GloVe embeddings and a 2-layer LSTM.

## Architecture
- **Embeddings**: GloVe 6B 100d (frozen or fine-tuned)
- **Encoder**: 2-layer LSTM (128 hidden, dropout 0.3)
- **Classifier**: Linear → Sigmoid

## Files
| File         | Purpose                                         |
| ------------ | ----------------------------------------------- |
| `data.py`    | Load SST-2 dataset                              |
| `vocab.py`   | Build vocab + GloVe embedding matrix            |
| `dataset.py` | PyTorch Dataset + DataLoader                    |
| `model.py`   | SentimentLSTM architecture                      |
| `train.py`   | Training loop with frozen/fine-tuned comparison |

## Setup
```bash
# Download GloVe
mkdir data && cd data
wget https://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip

# Run
python vocab.py      # builds vocab + embedding matrix
python train.py      # trains both phases