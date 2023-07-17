import sys

SYMBOL = '▁'

newVocab = set()
newVocabList = []

for line in open(sys.argv[1]):
    line = line.rstrip()
    token, score = line.split('\t')
    if token == SYMBOL:
        pass
    elif token.startswith(SYMBOL):
        token = token.replace(SYMBOL, '')
    if token not in newVocab:
        newVocab.add(token)
        newVocabList.append(token)

for line in newVocabList:
    print('%s\t0.0'%line)

#print('size of new vocab:', len(newVocab))