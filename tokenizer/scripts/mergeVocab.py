import sys

vocabSet = set()
vocabList = []
newLineFlag = False

for path in sys.argv[1:]:
    for line in open(path):
        if line == '\n':
            newLineFlag = True
            continue
        elif newLineFlag:
            token = '\n'
            newLineFlag = False
        else:
            token, score = line.rstrip().split('\t')
        if token not in vocabSet:
            vocabSet.add(token)
            vocabList.append(token)

for token in vocabList:
    print('%s\t0.0'%token)
