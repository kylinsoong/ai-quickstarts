from transformers import BertTokenizer, BertModel
import torch

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

sentence = "I heard a dog bark loudly at a cat"

inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)

with torch.no_grad():
    outputs = model(**inputs)

embeddings = outputs.last_hidden_state

sentence_embedding = torch.mean(embeddings, dim=1)

print(f"Shape of sentence embedding: {sentence_embedding.shape}")
print(f"Sentence embedding:\n{sentence_embedding}")


