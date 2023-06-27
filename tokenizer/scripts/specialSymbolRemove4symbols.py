# prefix付き記号からprefixを削除する

import re
import sys
import regex

SYMBOLS1='!"#$%&\'()*+,-./:;<=>?@[\]^_`{|}~'
SYMBOLS2='、。，．・：；？！゛゜´｀¨＾￣＿ヽヾゝゞ〃仝々〆〇ー―‐／＼～∥｜…‥‘’“”（）〔〕［］｛｝〈〉《》「」『』【】＋－±×÷＝≠＜＞≦≧∞∴♂♀°′″℃￥＄￠￡％＃＆＊＠§☆★○●◎◇◆□■△▲▽▼※〒→←↑↓〓∈∋⊆⊇⊂⊃∪∩∧∨￢⇒⇔∀∃∠⊥⌒∂∇≡≒≪≫√∽∝∵∫∬Å‰♯♭♪'

SYMBOLS = set(list(SYMBOLS1) + list(SYMBOLS2))
SYMBOL = '▁'

p = regex.compile(r'[^\p{Script=Hiragana}\p{Script=Katakana}ー\p{Script=Han}〇一二三四五六七八九a-zA-Z ]')


newVocab = set()
newVocabList = []

for line in open(sys.argv[1]):
    line = line.rstrip()
    token, score = line.split('\t')
    if token == SYMBOL:
        pass
    elif token.startswith(SYMBOL) and bool(p.search(token[1])):
        #print(token)
        token = token.replace(SYMBOL, '')
        #print('->', token)

    if token not in newVocab:
        newVocab.add(token)
        newVocabList.append(token)

for line in newVocabList:
    print('%s\t0.0'%line)
