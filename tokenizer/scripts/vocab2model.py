# convert vocab file to sp.model
import argparse
import sentencepiece.sentencepiece_model_pb2 as model

# see: https://github.com/google/sentencepiece/blob/31656da0c9cccfc47d4f0e69fc32d55faac3e1e9/python/add_new_vocab.ipynb

def discardPieces(tknzr):
    while len(tknzr.pieces) > 0:
        tknzr.pieces.pop()
    return tknzr

def createNewModel(baseModelPath):
    tknzr = model.ModelProto()
    
    if baseModelPath is not None:
        tknzr.ParseFromString(open(baseModelPath, "rb").read())
        tknzr = discardPieces(tknzr)
    return tknzr

def loadVocab(vocabPath):
    vocab = []
    newLineFlag = False
    for line in open(vocabPath):
        if newLineFlag:
            line = line.rstrip().split('\t')
            assert line[0]=='', 'broken vocab?'
            vocab.append(('\n', line[1]))
            newLineFlag = False
            continue
        if line=='\n':
            newLineFlag = True
        else:
            token, score = line.rstrip().split('\t')
            if token=='':
                # skip empty token
                continue
            vocab.append((token, score))
    #vocab = [line.rstrip().split('\t') for line in open(vocabPath)]
    return vocab

def setVocabToTokenizer(tknzr, vocab, args):
    special_tokens = args.special_tokens_csv.split(",")
    for token in vocab:
        newToken = model.ModelProto().SentencePiece()
        newToken.piece = token[0].encode('utf8')
        newToken.score = float(token[1])

        if token[0] in special_tokens:
            newToken.type = newToken.CONTROL
        if token[0] == args.unk_token:
            newToken.type = newToken.UNKNOWN
        if token[0].startswith('<0x') and token[0].endswith('>'):
            newToken.type = newToken.BYTE

        tknzr.pieces.append(newToken)
    return tknzr

def save(tknzr, path):
    with open(path, 'wb') as f:
        f.write(tknzr.SerializeToString())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', help='output path')
    parser.add_argument('-v', '--vocab', help='vocab file (tsv of piece\tscore)')
    parser.add_argument('-bm', '--baseModel', default=None ,
                        help='sp-model to use the same normalization. vocabulary is ignored.')
    parser.add_argument('-unk', '--unk-token', default='<unk>')
    parser.add_argument('-st', '--special-tokens-csv', default='<s>,</s>,<mask>,<pad>,<CLS>,<EOD>,<SEP>')
    args = parser.parse_args()

    tknzr = createNewModel(args.baseModel)
    vocab = loadVocab(args.vocab)

    tknzr = setVocabToTokenizer(tknzr, vocab, args)
    save(tknzr, args.output)

if __name__ == '__main__':
    main()
