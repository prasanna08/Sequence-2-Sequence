import torchtext
import torch
import numpy as np
import nltk
import json

nltk.download('punkt')

class BatchGenerator(object):
    def __init__(self, batch_size, src, tgt, tokenized=False, language='de'):
        self.tok = lambda x: [i for i in nltk.word_tokenize(x)]
        self.x = src
        self.y = tgt
        self.src = torchtext.data.Field(tokenize=self.tok, init_token='<sos>', eos_token='<eos>', lower=True, include_lengths=True)
        self.tgt = torchtext.data.Field(tokenize=self.tok, init_token='<sos>', eos_token='<eos>', lower=True)
        if not tokenized:
            self.x = [self.src.preprocess(i) for i in self.x]
            self.y = [self.tgt.preprocess(i) for i in self.y]
        self.src.build_vocab(self.x, min_freq=3)
        self.tgt.build_vocab(self.y, min_freq=3)
        self.batch_size = batch_size
        self._cursor = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.shuffle()
    
    def set_batch_size(self, new_size):
        self.batch_size = new_size
    
    def shuffle(self):
        pairs = [(i, j) for i, j in zip(self.x, self.y)]
        np.random.shuffle(pairs)
        self.x = [p[0] for p in pairs]
        self.y = [p[1] for p in pairs]
        return
    
    def __iter__(self):
        self._cursor = 0
        return self
    
    @classmethod
    def load_from_metadata(cls, batch_size, fname):
        f = open(fname, 'r')
        data = json.loads(f.read())
        f.close()
        return cls(batch_size, data['x'], data['y'], tokenized=True, language=data['language'])
        
    def store_metadata(self, fname):
        data = {
            'x': self.x,
            'y': self.y,
            'language': self.language
        }
        f = open(fname, 'w')
        f.write(json.dumps(data))
        f.close()

    def sort_on_length(self, batch_input, batch_output):
        idcs = torch.Tensor([i for i, j in sorted(enumerate(batch_input[1]), reverse=True, key=lambda x:x[1])]).long()
        sentances = batch_input[0][:, idcs]
        lengths = batch_input[1][idcs]
        batch_output = batch_output[:, idcs]
        return (sentances, lengths), batch_output
            
    def __next__(self):
        if self._cursor + self.batch_size >= len(self.x):
            rem = (self._cursor + self.batch_size) - len(self.x)
            batch_inputs = self.src.process(self.x[self._cursor:] + self.x[:rem], device=self.device)
            batch_outputs = self.tgt.process(self.y[self._cursor:] + self.y[:rem], device=self.device)
            batch_inputs, batch_outputs = self.sort_on_length(batch_inputs, batch_outputs)
            self._cursor = rem
            return batch_inputs, batch_outputs, True
        _pcursor = self._cursor
        batch_inputs = self.src.process(self.x[self._cursor:self._cursor+self.batch_size], device=self.device)
        batch_outputs = self.tgt.process(self.y[self._cursor:self._cursor+self.batch_size], device=self.device)
        batch_inputs, batch_outputs = self.sort_on_length(batch_inputs, batch_outputs)
        self._cursor = (self._cursor + self.batch_size) % len(self.x)
        return batch_inputs, batch_outputs, True if _pcursor >= self._cursor else False

    def next(self):
        return self.__next__()
