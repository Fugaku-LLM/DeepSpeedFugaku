import argparse
import sys
from multigram import lm, train
import sentencepiece as spm
import numpy as np
from tqdm import tqdm

def normalize(sp, text):
    return sp.decode(sp.encode(text))

def loadSP(path):
    print('LOAD SP MODEL:', path)
    if path is None:
        return None
    sp = spm.SentencePieceProcessor()
    sp.load(path)
    return sp

def loadData(path, sp):
    data = []
    print('LOAD DATA:', path)
    for line in tqdm(open(path, encoding='utf-8', errors='ignore')):
        if sp is not None:
            line = normalize(sp, line)
        line = '▁' + line.replace(' ', '▁')
        data.append(line)

        # limit size for debug
        #if len(data) > 100000:
        #    break
    return data

def prepareMLM(path):
    print('PREPARE MLM')
    vocab = [line.split('\t')[0] for line in open(path)]
    mlm = lm.MultigramLM()
    mlm.setVocabFromWordList(vocab)
    return mlm, vocab

def prepare(args):
    sp = loadSP(args.spModel)
    data = loadData(args.data, sp)
    mlm, originalVocab = prepareMLM(args.vocab)
    print('DONE PREPARATION!')
    return data, mlm, originalVocab

def selectTrain(mode):
    if mode=='viterbi':
        lmtrain = train.viterbiTrain
    elif mode=='viterbiStepWise':
        lmtrain = train.viterbiTrainStepWise
    elif mode=='viterbiBatch':
        lmtrain = train.viterbiTrainBatch
    else:
        # (mode=='EM')
        lmtrain = train.EMTrain
    return lmtrain

def mlm2tsv(mlm, originalVocab):
    # dump scores in the original order
    lines = []
    for token in originalVocab:
        score = np.log(mlm.theta[mlm.piece_to_id(token)])
        lines.append('%s\t%f\n'%(token, score))
    return lines

def saveTSV(path, tsv):
    with open(path, 'w') as f:
        for line in tsv:
            f.write(line)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-sp', '--spModel', default=None, help='sp model for normalization if required.')
    parser.add_argument('-v', '--vocab', help='vocab of tsv format for (token, score). score is not necessarily required.')
    parser.add_argument('-d', '--data', help='path to data for training')
    parser.add_argument('-o', '--output', help='path to output')
    parser.add_argument('-tm', '--trainingMode', choices=['viterbi','viterbiStepWise','viterbiBatch','EM'], default='EM')
    parser.add_argument('-me', '--maxEpoch', default=10, type=int)
    args = parser.parse_args()

    data, mlm, originalVocab = prepare(args)

    lmtrain = selectTrain(args.trainingMode)

    print('START TRAIN')
    mlm = lmtrain(mlm=mlm, data=data, maxIter=args.maxEpoch, proning=False)

    print('DUMP TSV')
    tsv = mlm2tsv(mlm, originalVocab)
    saveTSV(args.output, tsv)

if __name__ == '__main__':
    main()
