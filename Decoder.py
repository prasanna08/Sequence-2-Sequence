import torch
import Attention

class Decoder(torch.nn.Module):
    ATTN_MAP = {
        'DotProductAttn': Attention.DotProductAttn.hidden_scaled_attn,
        'MultiplicativeAttn':  Attention.MultiplicativeAttn,
        'AdditiveAttn':  Attention.AdditiveAttn
    }

    def __init__(self, hidden_size, vocab_size, embedding_dim, num_layers=1, bidirectional=False, attn_type='DotProductAttn'):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        # self.embedding = torch.nn.Embedding.from_pretrained(weights)
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.lstm = torch.nn.LSTM(embedding_dim, hidden_size, dropout=0.2, num_layers=num_layers, bidirectional=bidirectional)
        self.out_layer = torch.nn.Linear(hidden_size, vocab_size)
        self.log_softmax = torch.nn.LogSoftmax(dim=1)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.attn_type = attn_type
        self.attn, self.slf_attn, self.attn_proj = self.get_attn()

    def get_attn(self):
        attn = self.ATTN_MAP[self.attn_type]
        hidden = (2 if self.bidirectional else 1) * self.hidden_size
        return attn(hidden), attn(hidden), torch.nn.Linear(3*hidden, self.hidden_size)

    def forward(self, inputs, hidden, enc_output, prev_output):
        inputs = self.embedding(inputs).unsqueeze(dim=0)
        outputs, hidden = self.lstm(inputs, hidden)
        outputs = outputs[0]
        
        if prev_output is not None:
            prev_output = torch.cat([prev_output, outputs.unsqueeze(dim=0)], dim=0)
        else:
            prev_output = outputs.unsqueeze(dim=0)
        
        slf_cntxt, slf_attn_weights = self.slf_attn(prev_output, outputs)
        slf_cntxt = slf_cntxt.view(outputs.size())

        cntxt, attn_weights = self.attn(enc_output, outputs)
        cntxt = cntxt.view(outputs.size())
        
        outputs = self.attn_proj(torch.cat([cntxt, slf_cntxt, outputs], dim=1))
            
        outputs = self.log_softmax(self.out_layer(outputs))
        
        return outputs, hidden, attn_weights
    
    def init_hidden(self, batch_size):
        s = self.num_layers * (2 if self.bidirectional else 1)
        return [torch.zeros(size=(s, batch_size, self.hidden_size), device=self.device), torch.zeros(size=(s, batch_size, self.hidden_size), device=self.device)]
