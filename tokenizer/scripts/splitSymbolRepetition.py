import re
import sys

SYMBOLS1='!"#$%&\'()*+,-./:;<=>?@[\]^_`{|}~'
SYMBOLS2='、。，．・：；？！゛゜´｀¨＾￣＿ヽヾゝゞ〃仝々〆〇ー―‐／＼～∥｜…‥‘’“”（）〔〕［］｛｝〈〉《》「」『』【】＋－±×÷＝≠＜＞≦≧∞∴♂♀°′″℃￥＄￠￡％＃＆＊＠§☆★○●◎◇◆□■△▲▽▼※〒→←↑↓〓∈∋⊆⊇⊂⊃∪∩∧∨￢⇒⇔∀∃∠⊥⌒∂∇≡≒≪≫√∽∝∵∫∬Å‰♯♭♪'

SYMBOLS = SYMBOLS1 + SYMBOLS2


def replace(line, target, after):
    res = re.search('[%s][%s]'%(SYMBOLS, SYMBOLS), line)
    if res is None:
        return line

    i, j = res.span()
    neoline = line[:i+1] + ' ' + line[j-1:]

    if line==neoline:
        return line
    return replace(neoline, target, after)

f = open(sys.argv[1])
line = f.readline()

while line:
    line = line.strip()
    for s in SYMBOLS:
        line = replace(line, '%s%s'%(s, s), '%s %s'%(s, s))
    print(line)
    line = f.readline()
