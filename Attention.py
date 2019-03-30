import torch
import numpy as np

class DotProductAttn(torch.nn.Module):
    def __init__(self, hidden_size, scaling=1):
        """
        hidden_size = num_directions * hidden_size
        """
        super(DotProductAttn, self).__init__()
        self.hidden_size = hidden_size
        self.softmax = torch.nn.Softmax(dim=1)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaling = np.sqrt(scaling)
    
    @classmethod
    def hidden_scaled_attn(cls, hidden_size):
        """
        Scaled dot product attention with scaling factor same as hidden size
        of RNN Cell.
        """
        return cls(hidden_size, scaling=hidden_size)

    def forward(self, encoder_output, cur_hidden):
        """
        Dimensions:
        encoder_output: (time_step, batch_size, num_directions * hidden_size)
        cur_hidden: (batch_size, num_directions * hidden_size)
        """
        h1 = encoder_output.permute(1, 0, 2)
        h2 = cur_hidden.unsqueeze(dim=-1)
        attn_weights = self.softmax(torch.matmul(h1, h2) / self.scaling)
        cntxt = torch.matmul(attn_weights.permute(0, 2, 1), h1).squeeze()
        return cntxt, attn_weights.squeeze()

class MultiplicativeAttn(torch.nn.Module):
    def __init__(self, hidden_size):
        """
        hidden_size = num_directions * hidden_size
        """
        super(MultiplicativeAttn, self).__init__()
        self.hidden_size = hidden_size
        self.linear = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.softmax = torch.nn.Softmax(dim=1)
        
    def forward(self, encoder_output, cur_hidden):
        """
        Dimensions:
        encoder_output: (time_step, batch_size, num_directions * hidden_size)
        cur_hidden: (batch_size, num_directions * hidden_size)
        """
        h1 = self.linear(encoder_output.view(-1, self.hidden_size)).view(encoder_output.size())
        h1 = h1.permute(1, 0, 2)
        h2 = cur_hidden.unsqueeze(dim=-1)
        attn_weights = self.softmax(torch.matmul(h1, h2))
        cntxt = torch.matmul(attn_weights.permute(0, 2, 1), h1).squeeze()
        return cntxt, attn_weights.squeeze()

class AdditiveAttn(torch.nn.Module):
    def __init__(self, hidden_size):
        """
        hidden_size = num_directions * hidden_size
        """
        super(AdditiveAttn, self).__init__()
        self.hidden_size = hidden_size
        self.linear1 = torch.nn.Linear(2*self.hidden_size, self.hidden_size)
        self.tanh = torch.nn.Tanh()
        self.linear2 = torch.nn.Linear(self.hidden_size, 1)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def forward(self, encoder_output, cur_hidden):
        """Takes previous encoder output hidden states and current hidden state
        of decoder and outputs a context vector for current state of decoder.

        Dimensions:
        encoder_output: (time_step, batch_size, num_directions * hidden_size)
        cur_hidden: (batch_size, num_directions * hidden_size)
        """
        h1 = encoder_output
        attn_weights = torch.zeros(size=(h1.size()[0], h1.size()[1], 1), device=self.device)
        for i in range(0, h1.size()[0]):
            attn_weights[i] = self.linear2(self.tanh(self.linear1(torch.cat([h1[i], cur_hidden], dim=1))))
        attn_weights = self.softmax(attn_weights.permute(1, 2, 0))
        cntxt = torch.matmul(attn_weights, h1.permute(1, 0, 2)).squeeze()
        return cntxt, attn_weights.squeeze()
