import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import os
import sys
import random
import time

random.seed(123456789)
def generate_random_sentence(lexicon, length):
	sentence = ' '.join(random.choices(lexicon, k=length))
	return sentence.capitalize() + '.'


if __name__ == "__main__":

	batch_size = int(sys.argv[1])
	sentence_length = 128
	# Load the pre-trained DistilBERT model and tokenizer
	print("TimePreModelLoading --", time.time())
	model_name = 'distilbert-base-uncased'
	tokenizer = DistilBertTokenizer.from_pretrained(model_name)
	model = DistilBertForSequenceClassification.from_pretrained(model_name)

	# Set device to GPU if available, otherwise use CPU
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model.to(device)
	model.eval()
	print("TimePostModelLoading --", time.time())

	lexicon = ['The', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog']
	sentence_const = generate_random_sentence(lexicon, sentence_length)


	# Define the sentiment labels
	sentiment_labels = {
			0: 'Negative',
			1: 'Positive'
		}

	inputs = []
	for i in range(0, int(64 / batch_size)):
		input = [sentence_const] * batch_size
		# Tokenize input text
		input_sentence = tokenizer.batch_encode_plus(
			input,
			add_special_tokens=True,
			return_tensors='pt',
			truncation=True,
			padding='max_length',
			max_length = sentence_length
		)
		inputs.append(input_sentence)


	input_ids = input_sentence['input_ids'].to(device)
	attention_mask = input_sentence['attention_mask'].to(device)
	print("Generated Sentences")

	print("TimePreModel --", time.time())
	# Perform sentiment classification
	for i in range(0, 50):
		for input in inputs:
			outputs = model(input_ids, attention_mask=attention_mask)
	print("TimePostModel --", time.time())


