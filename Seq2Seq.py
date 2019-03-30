import torch
import time
import math

from Encoder import Encoder
from Decoder import Decoder

class Network(torch.nn.Module):
    def __init__(self, hidden_size, embedding_dim, src_vocab_size, tgt_vocab_size, sos_idx, eos_idx, pad_idx, num_layers=1, bidirectional=False, attn_type='DotProductAttn'):
        super(Network, self).__init__()
        # Set requires_grad = False below when using pretrained embeddings.
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.attn_type = attn_type
        self.encoder = Encoder(hidden_size, src_vocab_size, self.embedding_dim, self.num_layers, self.bidirectional)
        self.decoder = Decoder(hidden_size, tgt_vocab_size, self.embedding_dim, self.num_layers, self.bidirectional, self.attn_type)
        self.hidden_size = hidden_size
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.trainable_parameters = filter(lambda p: p.requires_grad, self.parameters())
        self.optimizer = torch.optim.Adam(self.trainable_parameters, lr=0.01)
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, 0.96)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.loss_fn = torch.nn.NLLLoss(ignore_index=pad_idx)
        self.initialize_params()
    
    def initialize_params(self):
        for p in self.parameters():
            if p.requires_grad and p.dim() > 1:
                torch.nn.init.xavier_normal_(p)
            if p.requires_grad and p.dim() == 1:
                torch.nn.init.normal_(p, mean=0, std=4e-2)

    def forward(self, i_seq, t_seq):
        # i_seq: (time_step, batch_size) shape.
        encoder_hidden = self.encoder.init_hidden(i_seq[0].size()[1])
        target_length = t_seq.size(0)

        encoder_outputs, encoder_hidden = self.encoder(i_seq, encoder_hidden)

        decoder_input = t_seq[0]
        decoder_outputs = torch.zeros(target_length-1, t_seq.size()[1], self.tgt_vocab_size)
        if self.bidirectional:
            decoder_hiddens = torch.zeros(target_length-1, t_seq.size()[1], 2*self.hidden_size)
        else:
            decoder_hiddens = torch.zeros(target_length-1, t_seq.size()[1], self.hidden_size)

        decoder_hidden = encoder_hidden

        for di in range(0, target_length-1):
            decoder_output, decoder_hidden, attn_weights = self.decoder(
                decoder_input, decoder_hidden, encoder_outputs, decoder_hiddens[:di].to(self.device) if di > 0 else None)
            topv, topi = decoder_output.topk(1)
            #decoder_input = torch.tensor([topi.squeeze().detach()], device=self.device)
            decoder_input = topi.squeeze().detach()
            decoder_outputs[di] = decoder_output

            last_decoder_hidden = decoder_hidden[0]
            if self.num_layers > 1:
                last_decoder_hidden = decoder_hidden[0].view(self.num_layers, 2, t_seq.size()[1], self.hidden_size)[-1]
            
            if self.bidirectional:
                decoder_hiddens[di] = last_decoder_hidden.permute(1, 0, 2).contiguous().view(-1, 2*self.hidden_size)
            else:
                decoder_hiddens[di] = last_decoder_hidden

            #if decoder_input.item() == EOS:
                #break

        return decoder_outputs
    
    def loss_compute(self, net_output, labels):
        net_output = net_output.to(torch.device('cpu'))
        labels = labels.to(torch.device('cpu'))
        loss = 0.0
        t_len = labels.size()[0]
        for i in range(1, t_len):
            loss += self.loss_fn(net_output[i-1], labels[i])
        return loss

    def train_iter(self, inp, outp):
        self.optimizer.zero_grad()
        net_output = self.forward(inp, outp)
        loss = self.loss_compute(net_output, outp)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 50.0)
        self.optimizer.step()
        return loss / outp.size()[0]

    def asMinutes(self, s):
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)

    def timeSince(self, since, percent):
        now = time.time()
        s = now - since
        es = s / (percent)
        rs = es - s
        return '%s (- %s)' % (self.asMinutes(s), self.asMinutes(rs))

    def trainer(self, bg, max_iters, summary_step=1, lr_decay_rate=3000, saver_steps=300):
        start = time.time()
        loss = 0.0
        epoch_count = 0
        step = 0
        for i in range(max_iters):
            x, y, epoch = bg.__next__()
            sloss = self.train_iter(x, y)
            loss += sloss.detach()
            step += 1
            if step % lr_decay_rate == 0:
                self.lr_scheduler.step()
            if step % summary_step == 0:
                #print(x.tolist(), y.tolist(), word_outs)
                print("Step: %d, Loss: %.4f, Time: %s" % (step, loss/step, self.timeSince(start, step/max_iters)))
            if step % saver_steps == 0:
                self.store_model(step, with_opt=True)
    
    def evaluate(self, i_seq, max_len=30):
        # i_seq: (time_step, batch_size) shape.
        i_sent = i_seq[0]
        i_len = i_seq[1]
        encoder_hidden = self.encoder.init_hidden(i_sent.size()[1])
        # Encoder output is output from last layer for all time steps.
        # Encoder hidden is hidden state and context vector of last time step for all layers.
        encoder_output, encoder_hidden = self.encoder((i_sent, i_len), encoder_hidden)

        # First input should be start of string token.
        decoder_input = torch.tensor([self.sos_idx], device=self.device)
        decoder_outputs = []
        if self.bidirectional:
            decoder_hiddens = torch.zeros(max_len, 1, 2*self.hidden_size, device=self.device)
        else:
            decoder_hiddens = torch.zeros(max_len, 1, self.hidden_size, device=self.device)

        decoder_hidden = encoder_hidden

        for i in range(0, max_len):
            decoder_output, decoder_hidden, attn_weights = self.decoder(decoder_input, decoder_hidden, encoder_output, decoder_hiddens[:i] if i > 0 else None)
            val, idx = decoder_output.topk(1)
            decoder_input = torch.tensor([idx.squeeze().detach()], device=self.device)
            decoder_outputs.append(idx.item())
            
            last_decoder_hidden = decoder_hidden[0]
            if self.num_layers > 1:
                last_decoder_hidden = decoder_hidden[0].view(self.num_layers, 2, 1, self.hidden_size)[-1]
            
            if self.bidirectional:
                decoder_hiddens[i] = last_decoder_hidden.permute(1, 0, 2).contiguous().view(-1, 2*self.hidden_size)
            else:
                decoder_hiddens[i] = last_dec
                oder_hidden

            if decoder_outputs[-1] == self.eos_idx:
                break

        return decoder_outputs
    
    def store_model(self, step=0, with_opt=False):
        data = {
            'network': net.state_dict(),
        }
        
        if with_opt:
            data['optimizer'] = net.optimizer.state_dict()
            data['lr'] = net.lr_scheduler.state_dict()
        torch.save(data, 'Seq2Seq-%d.pt' % (step))
        return
