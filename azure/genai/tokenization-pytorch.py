import torch
from torchtext.vocab import build_vocab_from_iterator
from torchtext.transforms import Sequential, VocabTransform, ToTensor
from torchtext.data.utils import get_tokenizer

# Example text for building the vocabulary
text = ["I heard a dog bark loudly at a cat"]

# Tokenizer to split the text
tokenizer = get_tokenizer("basic_english")

# Function to yield tokens for vocab building
def yield_tokens(data_iter):
    for text in data_iter:
        yield tokenizer(text)

# Build vocabulary from the text
vocab = build_vocab_from_iterator(yield_tokens(text), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])  # Handle out-of-vocabulary tokens

# Text transformations: tokenization -> vocab lookup -> tensor
text_transform = Sequential(
    VocabTransform(vocab),  # Converts tokens to indices
    ToTensor(padding_value=0, dtype=torch.long)  # Convert to tensor
)

# Input text to vectorize
test1 = "I heard a dog bark loudly at a cat"
test2 = "I heard a cat"

# Apply transformations to input texts
tokens1 = tokenizer(test1)
tokens2 = tokenizer(test2)

# Convert tokens to tensor representations
print(test1, text_transform(tokens1))
print(test2, text_transform(tokens2))

