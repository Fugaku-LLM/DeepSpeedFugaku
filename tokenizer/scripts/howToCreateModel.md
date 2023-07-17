# Tokenizer作成手順書
## データの準備
- 日本語
    - CC100-jaより1GBをダウンサンプリング
- 英語
    - CC100-enの先頭500000000行（6.8GB）のうち，1GBをサンプリング

## 前処理
- 日本語はMeCabで事前分割しておく
    - これは汚いトークンが語彙に含まれるのを防ぐため
    - 学習後にprefix_あり・なしの重複トークンを削除するため，実際に運用するときには事前分割不要
```
$ mecab -Owakati cc100ja_1GB.txt > cc100ja_1GB.txt.mecab
```
- 日本語，英語ともに記号の連続はスペース区切りに修正
漢字（含む中国語），平仮名，（半角・前核）片仮名，アルファベット以外はすべて区切っておく．
→これら以外の文字の連続は単語として認めない．

```
$ splitSymbols.py cc100ja_1GB.txt.mecab > cc100ja_1GB.txt.mecab.splitsymbols
$ splitSymbols.py cc100en_1GB.txt.mecab > cc100en_1GB.txt.splitsymbols
```


## SentencePieceモデルの作成
normalizationを行わない（`identity`設定）ことに注意．
- 特殊トークン（改行記号`<br>`など）は，`user_defined_symbols`として指定．
- 日本語 32K（重複削除分を考えると，40Kくらいとってもいいかも）
```
import sentencepiece as spm
spm.SentencePieceTrainer.train(
    input='/mnt/tamanegi/home1/tathi//tokenizer_training_data/cc100ja_1GB.txt.mecab.splitsymbols', 
    model_prefix='cc100ja40K_1GB', 
    vocab_size=40000, 
    num_threads=72,
    train_extremely_large_corpus=True,
    normalization_rule_name='identity',
    user_defined_symbols='<br>',
    max_sentencepiece_length=8,
    split_digits=True,
    byte_fallback=True
)
```
- 英語 18K （多めに20K）
```
import sentencepiece as spm
spm.SentencePieceTrainer.train(
    input='/mnt/tamanegi/home1/tathi//tokenizer_training_data/cc100en_1GB.txt.splitsymbols', 
    model_prefix='cc100en20K_1GB', 
    vocab_size=20000, 
    num_threads=72,
    train_extremely_large_corpus=True,
    normalization_rule_name='identity',
    user_defined_symbols='<br>',
    max_sentencepiece_length=16,
    split_digits=True,
    byte_fallback=True
)
```

## Tokenizerからprefix重複を削除
- 日本語
学習にはMeCab分割済みデータを使っているが，実際に利用するときはMeCabを挟まない．
そのため，prefix`▁`から始まるトークンはprefixを削除する．
重複がある場合は，片方のみ残す．

```
$ python specialSymbolRemove.py cc100ja32K_1GB.vocab > cc100ja32K_1GB.vocab.symbolRemoved
```

- 英語
英語の方も，記号類は前処理を挟んでいるので（上記前処理を参照），記号類だけprefixを削除する
```
$ python specialSymbolRemove4symbols.py cc100en20K_1GB.vocab > cc100en20K_1GB.vocab.symbolRemoved
```

## 語彙のマージ
重複している語彙は削除する．

```
$ python mergeVocab.py cc100ja_1GB/cc100ja40K_1GB.vocab.symbolRemoved cc100en20K_1GB.vocab.symbolRemoved > cc100ja40Ken20K.merged.vocab
```

## 手作業でのトークン選定
記号類は'_'とくっつける必要はないかもしれない（日本語の諸単語と同じ扱い）
記号連続をスペース区切りにしているため，不必要に_が多くなる．
再推定時や推論時にはnormalize等を施さないため，_から始まる記号トークンは廃止した方が良い．

## 再推定
- pythonで書いているので時間がかかる（2GBのデータで2.5時間程度）
- 再推定用にデータを作成
    - 前処理を行っていない日英コーパスを連結
```
$ cat cc100ja_1GB.txt cc100en_1GB.txt > cc100jaen_2GB.txt
```
- 再推定
```
$ python reestimateScore.py \
    -sp cc100ja32K_1GB.model \
    -v jaen.vocab \
    -d cc100jaen_2GB.txt \
    -o jaen.vocab.reestimated \
    -tm EM \
    -me 2
```
- データの読み込み＋SentencePieceModelによる前処理に時間がかかるので，処理済みのデータを`-d`オプションにpickle形式で渡すこともできる．
    - 末尾`.pkl`のpathを渡せばOK
- 並列処理が可能な環境であれば`-tm EMMT`とすることで，ある程度高速にEMアルゴリズムの学習が可能です．

## model化

```
$ python vocab2model.py \
    -v jaen.vocab.reestimated \
    -o jaen.vocab.reestimated.vocab \
    -bm cc100ja32K_1GB.model
```
