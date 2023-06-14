# オリジナルのvocabは精度が低いので，モデルの数値をそのままdump
import argparse
import sentencepiece as spm

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model')
parser.add_argument('-o', '--output')
args = parser.parse_args()

sp = spm.SentencePieceProcessor()
sp.load(args.model)

vocab = ['%s\t%s'%(str(sp.id_to_piece(i)), str(sp.get_score(i))) for i in range(sp.get_piece_size())]

with open(args.output, 'w') as f:
    for line in vocab:
        f.write(line+'\n')