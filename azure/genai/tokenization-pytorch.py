from transformers import BertTokenizer
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

sentence = "I heard a dog bark loudly at a cat"

inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)

print(inputs.input_ids)



