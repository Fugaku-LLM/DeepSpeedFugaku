# スペース連続とspecial tokenのスコアを0.0にする
# special tokensは先頭n個と指定する

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-v', '--vocab')
parser.add_argument('-nst', '--numSpecialTokens', type=int,
                    help='scores of first n tokens are replaced with 0.0')
parser.add_argument('--spaceNegativeOne', action='store_true')
args = parser.parse_args()

vocab = []
newLineFlag = False
for line in open(args.vocab):
    if line == '\n':
        newLineFlag = True
        continue
    elif newLineFlag:
        token = '\n'
        score = line.rstrip().split('\t')[1]
        newLineFlag = False
    else:
        token, score = line.rstrip().split('\t')
    vocab.append((token, score))

for i, v in enumerate(vocab):
    token, score = v
    if i < args.numSpecialTokens:
        if args.spaceNegativeOne and '▁' in token:
            score = '-1.0'
        else:
            score = '0.0'
    print('%s\t%s'%(token, score))    
