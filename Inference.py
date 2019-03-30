import argparse
import torch
import nltk

from Seq2Seq import Network
from BatchGenerator import BatchGenerator

parser = argparse.ArgumentParser()
parser.add_argument('--embedding-dim', help='Specify embedding dimesnion', default=300 ,type=int)
parser.add_argument('--hidden-size', help='Size of hidden layer', default=256 ,type=int)
parser.add_argument('--num-layers', help='Specify number of layers', default=2, type=int)
parser.add_argument('--bidirectional', help='Use bi-directional RNN model', default=True, type=bool)
parser.add_argument('--attention', help='Attention type to use', default='MultiplicativeAttn', type=str)
parser.add_argument('--model', help='Path to stored model', type=str)
parser.add_argument('--infile', help='Input file which contains english sentances on every line', type=str)
parser.add_argument('--tgtfile', help='Input file which contains target sentances on every line', type=str)
parser.add_argument('--outfile', help='Output file where translated sentances will be stored', type=str)
parser.add_argument('--language', help='Language to translate to: hi or de', default='de', type=str)

def evaluate(net, src, tgt, bg):
    inp = bg.src.process([src], device=net.device)
    out = net.evaluate(inp)
    out = [bg.tgt.vocab.itos[idx] for idx in out]
    score = nltk.translate.bleu_score.sentence_bleu(out, tgt)
    return out, score

def load_from_checkpoint(net, fname, with_opt=False):
    checkpoint = torch.load(fname, map_location='cpu')
    net.load_state_dict(checkpoint['network'])
    if with_opt:
        net.optimizer.load_state_dict(checkpoint['optimizer'])
        net.lr_scheduler.load_state_dict(checkpoint['lr'])
    return net

def evaluate_list(net, src_list, tgt_list, bg):
    outs = []
    scores = 0.0
    net.eval()
    for i in range(len(src_list)):
        out, score = evaluate(net, src_list[i], tgt_list[i], bg)
        outs.append(out)
        scores += score
    return outs, scores / len(src_list)

def evaluate_file(net, infile, tgtfile, outfile, bg):
    net.eval()
    fin = open(infile, 'r')
    sentances = list(fin.readlines())
    fin.close()
    ftgt = open(tgtfile, 'r')
    targets = list(ftgt.readlines())
    ftgt.close()
    src_tokenized = [bg.src.preprocess(x) for x in sentances]
    tgt_tokenized = [bg.tgt.preprocess(x) for x in targets]
    output_tokenized, bl_score = evaluate_list(net, src_tokenized, tgt_tokenized, bg)
    fout = open(outfile, 'w')
    outputs = [' '.join(x) for x in output_tokenized]
    fout.write('\n'.join(outputs))
    fout.close()
    return bl_score

if __name__=='__main__':
    args = parser.parse_args()
    
    if args.language == 'de':
        bg = BatchGenerator.load_from_metadata(10, './data/EN-DE-2-30.json')
    elif args.language == 'hi':
        bg = BatchGenerator.load_from_metadata(10, './data/EN-HI-2-30.json')
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
    net = net.to(net.device)
    net = load_from_checkpoint(net, args.model)
    score = evaluate_file(net, args.infile, args.tgtfile, args.outfile, bg)
    print('Overall BLEU Score is %.3f' % score)
    