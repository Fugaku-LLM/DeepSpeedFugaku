import argparse
import sentencepiece as spm

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data')
parser.add_argument('-vs', '--vocabSize', type=int)
parser.add_argument('-ml', '--maxLength', type=int)
parser.add_argument('-p', '--prefix')
parser.add_argument('-sbw', '--split-by-whitespace', action='store_true')
args = parser.parse_args()

spm.SentencePieceTrainer.train(
    input=args.data,
    model_prefix=args.prefix,
    vocab_size=args.vocabSize, 
    num_threads=72,
    train_extremely_large_corpus=True,
    normalization_rule_name='identity',
    user_defined_symbols=['\n','<pad>', '<CLS>', '<SEP>', '<EOD>', '<MASK>'], # ' '
    max_sentencepiece_length=args.maxLength,
    split_digits=True,
    byte_fallback=True,
    split_by_whitespace=args.split_by_whitespace, # whitespaceで事前分割しない
    allow_whitespace_only_pieces=True, # whitespaceトークンを許可
    remove_extra_whitespaces=False, # space連続の除去を不許可
)
