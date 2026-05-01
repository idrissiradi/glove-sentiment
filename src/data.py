# data.py
from datasets import load_dataset

dataset = load_dataset("stanfordnlp/sst2")

print(dataset)
print(dataset["train"][0])
print(dataset["validation"][0])
