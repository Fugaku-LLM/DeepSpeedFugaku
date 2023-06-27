import argparse
import sentencepiece as spm

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data')
parser.add_argument('-vs', '--vocabSize', type=int)
parser.add_argument('-ml', '--maxLength', type=int)
parser.add_argument('-p', '--prefix')
args = parser.parse_args()

spm.SentencePieceTrainer.train(
    input=args.data,
    model_prefix=args.prefix,
    vocab_size=args.vocabSize, 
    num_threads=72,
    train_extremely_large_corpus=True,
    normalization_rule_name='identity',
    user_defined_symbols='<br>',
    max_sentencepiece_length=args.maxLength,
    split_digits=True,
    byte_fallback=True
)
