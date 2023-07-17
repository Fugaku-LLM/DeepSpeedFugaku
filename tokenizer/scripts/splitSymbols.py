import regex
import sys
import re

# 漢字（中国語含む），ひらがな，カタカナ，アルファベット以外は全てsplitする
#text = 'こんにちは sorry!!!「」'
p = regex.compile(r'[^ ][^\p{Script=Hiragana}\p{Script=Katakana}ー\p{Script=Han}〇一二三四五六七八九a-zA-Z ]')
#print(p.search(text))

def replace(line):
    res = p.search(line)
    if res is None:
        return line

    i, j = res.span()
    #neoline = line[:i+1] + ' ' + line[j-1:]
    if j==len(line):
        neoline = line[:i+1] + ' ' + line[j-1:]
    else:
        neoline = line[:i+1] + ' ' + line[j-1] + ' ' + line[j:]

    if line==neoline:
        return line
    return replace(neoline)

f = open(sys.argv[1])
line = f.readline()

while line:
    line = line.rstrip()
    line = replace(line)
    print(line)
    line = f.readline()
