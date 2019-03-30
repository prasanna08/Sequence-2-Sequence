import torch

class Encoder(torch.nn.Module):
    def __init__(self, hidden_size, vocab_size, embedding_dim, num_layers=1, bidirectional=False):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        # self.embedding = torch.nn.Embedding.from_pretrained(weights)
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.lstm = torch.nn.LSTM(embedding_dim, hidden_size, dropout=0.2, num_layers=num_layers, bidirectional=bidirectional)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def forward(self, inputs, hidden):
        ips, lengths = inputs
        inputs = self.embedding(ips)
        packed_embedded = torch.nn.utils.rnn.pack_padded_sequence(inputs, lengths)
        outputs, hidden = self.lstm(packed_embedded, hidden)
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        return outputs, hidden
    
    def init_hidden(self, batch_size):
        s = self.num_layers * (2 if self.bidirectional else 1)
        return (torch.zeros(size=(s, batch_size, self.hidden_size), device=self.device), torch.zeros(size=(s, batch_size, self.hidden_size), device=self.device))
