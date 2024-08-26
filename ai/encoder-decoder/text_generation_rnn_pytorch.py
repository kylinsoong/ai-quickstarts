import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random

# Define the Encoder
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size)

    def forward(self, input):
        embedded = self.embedding(input)
        output, hidden = self.rnn(embedded)
        return hidden

# Define the Decoder
class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        input = input.unsqueeze(0)
        embedded = self.embedding(input)
        output, hidden = self.rnn(embedded, hidden)
        output = self.out(output.squeeze(0))
        return output, hidden

# Define the Seq2Seq Model
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src: source sequence
        # trg: target sequence
        # teacher_forcing_ratio: probability of using teacher forcing

        trg_len = trg.shape[0]
        batch_size = trg.shape[1]
        trg_vocab_size = self.decoder.out.out_features

        # tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(device)

        # encode the input sequence
        hidden = self.encoder(src)

        # first input to the decoder is the <sos> token
        input = trg[0, :]

        for t in range(1, trg_len):
            output, hidden = self.decoder(input, hidden)
            outputs[t] = output

            # decide if we will use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio

            # get the highest predicted token from softmax output
            top1 = output.argmax(1)

            # if teacher forcing, use actual next token as next input; otherwise, use predicted token
            input = trg[t] if teacher_force and t < trg_len else top1

        return outputs

device = torch.device("cpu")
if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")

# Example usage
input_size = 10  # size of the input vocabulary
output_size = 10  # size of the output vocabulary
hidden_size = 256  # size of the hidden state in the encoder and decoder
learning_rate = 0.001
epochs = 5

encoder = Encoder(input_size, hidden_size)
decoder = Decoder(hidden_size, output_size)
model = Seq2Seq(encoder, decoder).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Dummy data
src = torch.randint(0, input_size, (5, 32)).to(device)  # source sequence
trg = torch.randint(0, output_size, (7, 32)).to(device)  # target sequence

# Training loop
for epoch in range(epochs):
    optimizer.zero_grad()
    output = model(src, trg)
    
    # Flatten the output and target tensors for the loss calculation
    loss = criterion(output.view(-1, output_size), trg.view(-1))
    
    loss.backward()
    optimizer.step()

    print(f'Epoch: {epoch+1}/{epochs}, Loss: {loss.item()}')
