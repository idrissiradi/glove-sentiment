# Sentiment Analysis with GloVe + LSTM

A binary sentiment classifier that reads movie reviews and predicts whether they're positive or negative.
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
| `predict.py` | Inference on custom text                        |


## Setup
```bash
# 1. Download GloVe
mkdir data && cd data
wget https://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip
cd ..

# 2. Build vocabulary and embedding matrix
python vocab.py

# 3. Train model (both frozen and fine-tuned phases)
python train.py

# 4. Predict on your own text
python predict.py
```

## Results

| Phase                                 | Val Accuracy |
| ------------------------------------- | ------------ |
| Frozen embeddings (transfer learning) | 80.69%       |
| Fine-tuned embeddings                 | 81.81%       |

Fine-tuning gives +1.1% but shows overfitting signs (train acc 94% vs val acc 81%).
Frozen embeddings are safer for this dataset size.