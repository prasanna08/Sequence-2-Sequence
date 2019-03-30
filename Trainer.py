import argparse
import torch

from Seq2Seq import Network
from BatchGenerator import BatchGenerator

parser = argparse.ArgumentParser()
parser.add_argument('--embedding-dim', help='Specify embedding dimesnion', default=300 ,type=int)
parser.add_argument('--hidden-size', help='Size of hidden layer', default=256 ,type=int)
parser.add_argument('--num-layers', help='Specify number of layers', default=4, type=int)
parser.add_argument('--iters', help='Specify maximum iterations', default=100000, type=int)
parser.add_argument('--bidirectional', help='Use bi-directional RNN model', default=True, type=bool)
parser.add_argument('--batch-size', help='Specify batch size', default=100, type=int)
parser.add_argument('--summary-steps', help='Display Summary at every n iterations', default=10, type=int)
parser.add_argument('--checkpoint-steps', help='Save checkpoints at every n steps of minibatch', default=500, type=int)
parser.add_argument('--attention', help='Attention type to use', default='DotProductAttn', type=str)
parser.add_argument('--language', help='Language to translate to: hi or de', default='de', type=str)

if __name__ == '__main__':
    args = parser.parse_args()
    
    if args.language == 'de':
        bg = BatchGenerator.load_from_metadata(args.batch_size, './data/EN-DE-2-30.json')
    elif args.language == 'hi':
        bg = BatchGenerator.load_from_metadata(args.batch_size, './data/EN-HI-2-30.json')
    else:
        raise Exception('%s language is not available. Use either hi or de.')
    
    embedding_dim = args.embedding_dim
    hidden_size = args.hidden_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    src_vocab_size = len(bg.src.vocab)
    tgt_vocab_size = len(bg.tgt.vocab)
    sos_idx = bg.tgt.vocab.stoi['<sos>']
    eos_idx = bg.tgt.vocab.stoi['<eos>']
    pad_idx = bg.tgt.vocab.stoi['<pad>']
    net = Network(hidden_size, embedding_dim, src_vocab_size, tgt_vocab_size, sos_idx, eos_idx, pad_idx, args.num_layers, args.bidirectional, args.attention)
    net = net.to(net.device).train()
    net.trainer(bg, max_iters=args.iters, summary_step=args.summary_steps, saver_steps=args.checkpoint_steps)
