import torch
from transformers import BertTokenizer, BertForSequenceClassification
import random
import string
import numpy as np
import time
import sys

random.seed(123456789)

def generate_random_sentence(size_kb):
    target_size_bytes = size_kb * 1024
    avg_bytes_per_char = 1

    num_chars = target_size_bytes // avg_bytes_per_char
    sentence = ''.join(random.choice(string.ascii_lowercase + string.ascii_uppercase + ' ') for _ in range(int(num_chars)))
    return sentence.strip()


if __name__ == "__main__":
    batch_size = int(sys.argv[1])
    print("TimePreModelLoading --", time.time())
    model_name = "prajjwal1/bert-tiny"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name)
    model.to(device)
    model.eval()
    print("TimePostModelLoading --", time.time())

    sentences = []
    np.random.seed(123456789)
    sentence_const = generate_random_sentence(size_kb=1.1)
    count = 64
    print("Generated Sentences")

    inputs = []
    for i in range(0, int(64 / batch_size)):
        input = [sentence_const] * batch_size
        inputs.append(tokenizer(input, padding=True, truncation=True, return_tensors="pt").to(device))

    print("TimePreModel --", time.time())
    for i in range(0, 50):
        for input in inputs:
            outputs = model(**input)
    print("TimePostModel --", time.time())
