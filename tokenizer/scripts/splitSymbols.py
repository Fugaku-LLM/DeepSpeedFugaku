import regex
import sys
import re
import threading

# 再帰の深さ上限を多めにとっておく
sys.setrecursionlimit(67108864) #64MB
threading.stack_size(1024*1024)  #2の20乗のstackを確保=メモリの確保

# 漢字（中国語含む），ひらがな，カタカナ，アルファベット以外は全てsplitする
#text = 'こんにちは sorry!!!「」'
p1 = regex.compile(r'[^ ][^\p{Script=Hiragana}\p{Script=Katakana}ー\p{Script=Han}〇一二三四五六七八九a-zA-Z ]')
p2 = regex.compile(r'[^\p{Script=Hiragana}\p{Script=Katakana}ー\p{Script=Han}〇一二三四五六七八九a-zA-Z ][^ ]')
#print(p.search(text))

def replaceForward(line):
    res = p1.search(line)
    if res is None:
        return line

    i, j = res.span()
    neoline = line[:i+1] + ' ' + line[j-1:]
    #if j==len(line):
    #    neoline = line[:i+1] + ' ' + line[j-1:]
    #else:
    #    neoline = line[:i+1] + ' ' + line[j-2] + ' ' + line[j-1:]

    if line==neoline:
        return line
    return replaceForward(neoline)

def replaceBackward(line):
    res = p2.search(line)
    if res is None:
        return line

    i, j = res.span()
    neoline = line[:i+1] + ' ' + line[j-1:]
    #if j==len(line):
    #    neoline = line[:i+1] + ' ' + line[j-1:]
    #else:
    #    neoline = line[:i+1] + ' ' + line[j-2] + ' ' + line[j-1:]

    if line==neoline:
        return line
    return replaceBackward(neoline)

f = open(sys.argv[1])
line = f.readline()

i = 0
while line:
    line = line.rstrip()

    if 1000 < len(line):
        # 1000文字以上の行は1000文字で切り捨て
        line = line[:1000]

    line = replaceForward(line) 
    line = replaceBackward(line)  
    print(line)
    line = f.readline()

