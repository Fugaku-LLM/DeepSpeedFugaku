import sys

vocabSet = set()
vocabList = []

for path in sys.argv[1:]:
    for line in open(path):
        token, score = line.rstrip().split('\t')
        if token not in vocabSet:
            vocabSet.add(token)
            vocabList.append(token)

for token in vocabList:
    print('%s\t0.0'%token)
