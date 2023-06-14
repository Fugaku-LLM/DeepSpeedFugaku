import argparse
import sentencepiece as spm
import numpy as np
from collections import Counter

def calcPPLwithNewUnigram(encodedData):
    unigramCount = Counter(encodedData)
    total = len(encodedData)
    unigramDict = {k:np.log(v/total)
                    for k, v in unigramCount.items()}
    neglogs = [unigramDict[idx] for idx in encodedData]
    entropy = - sum(neglogs) / len(neglogs)
    ppl = np.exp(entropy)

    return ppl

def calcPPLwithSP(tknzr, encodedData):
    neglogs = [tknzr.get_score(idx) for idx in encodedData]

    entropy = - sum(neglogs) / len(neglogs)
    ppl = np.exp(entropy)

    return ppl

def loadData(args):
    print('DATA:', args.data)
    data = [line.strip() for line in open(args.data)]
    return data

def loadTokenizer(args):
    print('TOKENIZER:', args.tokenizer)
    sp = spm.SentencePieceProcessor()
    sp.load(args.tokenizer)
    print('VOCAB SIZE:', sp.vocab_size())
    return sp

def encodeData(tknzr, data):
    encodedData = [idx for line in data 
                       for idx in tknzr.encode_as_ids(line)]
    print('# ORIGINAL TOKENS:', len(encodedData))

    # filter special tokens
    idxset = set([tknzr.bos_id(), tknzr.eos_id(), 
                  tknzr.pad_id(), tknzr.unk_id()])
    filteredData = [idx for idx in encodedData
                        if idx not in idxset]
    print('# TOKENS AFTER FILTERING:', len(filteredData))
    print('\t# Ignored BOS:', encodedData.count(tknzr.bos_id()))
    print('\t# Ignored EOS:', encodedData.count(tknzr.eos_id()))
    print('\t# Ignored PAD:', encodedData.count(tknzr.pad_id()))
    print('\t# Ignored UNK:', encodedData.count(tknzr.unk_id()))

    print('# USED TOKEN TYPES:', len(set(encodedData)))

    return filteredData

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--tokenizer', help='path to sentencepiece model of **UNIGRAM**')
    parser.add_argument('-d', '--data', help='path to data for evaluation')
    args = parser.parse_args()

    tknzr = loadTokenizer(args)
    data = loadData(args)

    encodedData = encodeData(tknzr, data)

    pplsp = calcPPLwithSP(tknzr, encodedData)
    print('PPL (SP MODEL): %.2f'%pplsp)

    ppluni = calcPPLwithNewUnigram(encodedData)
    print('PPL (NEW UNIGRAM) %.2f'%ppluni)

if __name__=='__main__':
    main()