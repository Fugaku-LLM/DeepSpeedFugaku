# 使い方
## SentencePieceモデルの評価（ユニグラム）
テキスト上でのトークナイズのPerplexityを計測します．
JGLUEで配布されているデータのうち，training splitをサンプルとして`DeepSpeedFugaku/tokenizer/data`に同梱しています．


```
$  python evalPPL.py \
    -t path/to/sp.model \
    -d ../data/jsts-v1.1-train.txt
```

出力例：
```
TOKENIZER: models/cc100ja_1GB_mecab/sp.model
VOCAB SIZE: 32000
DATA: ../data/jglue_rawtexts/jsts-v1.1-train.txt
# ORIGINAL TOKENS: 350108
# TOKENS AFTER FILTERING: 348568
        # Ignored BOS: 0
        # Ignored EOS: 0
        # Ignored PAD: 0
        # Ignored UNK: 1540
# USED TOKEN TYPES: 5486
PPL (SP MODEL): 1286.86
PPL (NEW UNIGRAM) 275.65
```

`PPL (SP MODEL)`は，SentencePieceが持つユニグラム言語モデルのスコアをもとに計測したPerplexityです．
`PPL (NEW UNIGRAM)`は，モデルでテキストを分割してから，新たに頻度数え上げでユニグラム言語モデルを作成し，計測したPerplexityです．
新たに作成したユニグラム言語モデルの語彙の規模は`# USED TOKEN TYPE`に記載されています．
なお，未知トークンは計測対象から除外されています．

## モデルの作成
SentencePieceのモデル作成を前提としています．
（細かい手順は，後ほどまとめます）
各モジュールの使い方を列挙します．

### vocab2model.py
SentencePieceのモデル作成時に出力された`.vocab`ファイルから，SentencePieceモデルとして読み込み可能な`.model`ファイルを作成します．
語彙やスコアを外部ツールでチューニングする場合は，`.vocab`の形式にしておくことでSentencePieceで読み込み可能なモデルに変換できます．

```
$ python vocab2model.py \
    -v path/to/vocab \
    -o path/to/output.model \
    -bm path/to/original.model
```

オリジナルのSentencePieceモデルと同じ正規化等の設定を利用するために，`.vocab`ファイルと同時に作成されているオリジナルの`.model`ファイルを`-bm`オプションで与えることを推奨します．
`<x0`から始まるトークンは，byte-fallback用のタグが自動で振られます．

### model2vocab.py
`vocab2model.py`の逆操作です．
`.vocab`ファイルを紛失している場合に利用します．

### specialSymbolRemove.py
SentencePieceが自動で付与するprefix `▁`を`.vocab`から削除するために使います．
`▁abc`と`abc`のようにprefixの有無以外に差が無いトークンを後者に統一するために使用します．

### reestimateScore.py
`.vocab`ファイルと学習データを受け取り，各トークンのスコアを再推定するためのスクリプトです．

この処理は，平岡が作成した以下のリポジトリに依存しますが，`scripts/multigram`以下に同梱しています．
必要であれば以下の手順で`pip install`を行ってください．
```
(必要な場合のみ)
$ cd tokenizer/scripts/multigram
$ pip install --editable .
```

再推定は以下のコマンドで実行できます．
```
$ python reestimateScore.py \
    -sp path/to/original.model \
    -v path/to/vocab \
    -d path/to/data \
    -o path/to/output \
    -tm EM \
    -me 2
```

`-v`では`.vocab`ファイル形式の語彙を渡します．`token, score`が並ぶTSV形式ですが，scoreの値は適当で構いません．

`-d`では，推定に使用する学習データを指定します．
オリジナルのSentencePieceモデルと同じ正規化を用いる場合は，`-sp`にオリジナルのSentencePieceモデルを指定します．
（データ自体に`spm_normalize`を施している場合は不要です）

`-tm`では推定に用いるアルゴリズムを指定します．特に理由が無ければ`EM`を指定します．
`-me`はアルゴリズムの最大学習エポックです．こちらも理由が無ければ`2`で十分です．


再推定の性能は以下の通りです．
（`evalPPL.py`を用いてJSTSデータで評価）

CC100-jaから1GBをダウンサンプリングして作成したオリジナルのSentencePieceモデル：
```
TOKENIZER: models/cc100ja_1GB_mecab/cc100ja32K_1GB_mecab.model
VOCAB SIZE: 32000
DATA: ../data/jsts-v1.1-train.txt
# ORIGINAL TOKENS: 350280
# TOKENS AFTER FILTERING: 348740
        # Ignored BOS: 0
        # Ignored EOS: 0
        # Ignored PAD: 0
        # Ignored UNK: 1540
# USED TOKEN TYPES: 5479
PPL (SP MODEL): 1204.34
PPL (NEW UNIGRAM) 274.00
```

上記モデルのvocabファイルと，1GBの同じ学習データを用いてスコアを再推定したモデル：
(上記結果とほぼ同じになるはず)
```
TOKENIZER: models/cc100ja_1GB_mecab/cc100ja32K_1GB_mecab.vocab.reestimated_full_em2ep.model
VOCAB SIZE: 32000
DATA: ../data/jsts-v1.1-train.txt
# ORIGINAL TOKENS: 350108
# TOKENS AFTER FILTERING: 348568
        # Ignored BOS: 0
        # Ignored EOS: 0
        # Ignored PAD: 0
        # Ignored UNK: 1540
# USED TOKEN TYPES: 5486
PPL (SP MODEL): 1286.86
PPL (NEW UNIGRAM) 275.65
```

PPLはほぼ同等であり，推定結果としては妥当．
EMアルゴリズムは確率的な振る舞いをするため，必ずしも完全一致しない．
なお，再推定アルゴリズムは並列化や高速化に力を入れていないため，かなり時間がかかる．
