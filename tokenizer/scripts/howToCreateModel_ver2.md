# Tokenizer作成手順書
## データの準備
NII-LLMで共有されているコーパス v1を利用．
- 日本語 (`ja.txt`)
    - ja_ccから1GBを乱択
    - ja_wikiから1GBを乱択
- 英語 (`en.txt`)
    - en_wikiから1GBを乱択
    - en_pileから1GBを乱択
- コード (`code.txt`)
    - stackから1GBを乱択

（合計5GB程度）

## 前処理
- 日本語はMeCabで事前分割しておく
    - これは汚いトークンが語彙に含まれるのを防ぐため
    - 学習後にprefix_あり・なしの重複トークンを削除するため，実際に運用するときには事前分割不要
    - 1行が非常に長いデータがあるため `-b`オプションを多めにとっておく．
    ```
    # （長すぎるとmecabの解析に時間がかかるので区切っておくと扱いやすい）
    $ sed "s/。/。\n/g" ja.txt > ja.txt.newline
    $ mecab -Owakati -b 100000000 ja.txt.newline > ja.txt.newline.mecab
    ```
- 日本語，英語ともに記号の連続はスペース区切りに修正
漢字（含む中国語），平仮名，（半角・前核）片仮名，アルファベット以外はすべて区切っておく．
→これら以外の文字の連続は単語として認めない．
    ```
    $ splitSymbols.py ja.txt.newline.mecab > ja.txt.newline.mecab.splitsymbols
    $ splitSymbols.py en.txt > en.txt.splitsymbols
    $ splitSymbols.py code.txt > code.txt.splitsymbols
    ```


## SentencePieceモデルの作成
normalizationを行わない（`identity`設定）ことに注意．
- `\n`以外の特殊トークン（`<pad>`など）は後で追加するので，ここで指定しなくてもよい．
- 日本語（例：30K）
    ```
    spm.SentencePieceTrainer.train(
        input=ja.txt.newline.mecab.splitsymbols,
        model_prefix=ja_30K,
        vocab_size=30000,
        num_threads=72,
        train_extremely_large_corpus=True,
        normalization_rule_name='identity',
        user_defined_symbols=['\n'],
        max_sentencepiece_length=8, # 日本語は最大長8
        split_digits=True,
        byte_fallback=True,
        split_by_whitespace=True, # モデル作成時は空白で区切る
        allow_whitespace_only_pieces=True, 
        remove_extra_whitespaces=False, 
    )
    ```
- 英語（例：20K）
    ```
    spm.SentencePieceTrainer.train(
        input=en.txt.splitsymbols,
        model_prefix=en_20K,
        vocab_size=20000,
        num_threads=72,
        train_extremely_large_corpus=True,
        normalization_rule_name='identity',
        user_defined_symbols=['\n'],
        max_sentencepiece_length=8, # 英語・コードは最大長16
        split_digits=True,
        byte_fallback=True,
        split_by_whitespace=True, # モデル作成時は空白で区切る
        allow_whitespace_only_pieces=True, 
        remove_extra_whitespaces=False, 
    )
    ```
- コード（例：10K）
    ```
    spm.SentencePieceTrainer.train(
        input=code.txt.splitsymbols,
        model_prefix=en_10K,
        vocab_size=10000,
        num_threads=72,
        train_extremely_large_corpus=True,
        normalization_rule_name='identity',
        user_defined_symbols=['\n'],
        max_sentencepiece_length=8, # 英語・コードは最大長16
        split_digits=True,
        byte_fallback=True,
        split_by_whitespace=True, # モデル作成時は空白で区切る
        allow_whitespace_only_pieces=True, 
        remove_extra_whitespaces=False, 
    )
    ```

同梱の `createModel.py` を使う場合は以下のようなコマンドを叩く．

```
python /home/tathi/work/tokenizer/createModel.py \
    --data ja_ja.txt.newline.mecab.splitsymbols \
    --prefix ja_30K \
    --vocabSize 30000 \
    --maxLength 8 \
    --split-by-whitespace 
```


## Tokenizerからprefix重複を削除
- 日本語
    - 学習にはMeCab分割済みデータを使っているが，実際に利用するときはMeCabを挟まない．
    - そのため，prefix`▁`から始まるトークンはprefixを削除する．
    - 重複がある場合は，片方のみ残す．

    ```
    $ python specialSymbolRemove.py ja_30K.vocab > ja_30K.vocab.symbolRemoved
    ```

- 英語・コード
    - 英語やコードのモデルも，記号類は前処理を挟んでいるので（上記前処理を参照），記号類だけprefixを削除する
    ```
    $ python specialSymbolRemove4symbols.py en_20K.vocab > en_20K.vocab.symbolRemoved
    $ python specialSymbolRemove4symbols.py code_10K.vocab > code_10K.vocab.symbolRemoved
    ```

## 語彙のマージ
- 上記で作成したファイルと，役物語彙を列挙したファイルをマージする．
    - **`specialTokens.vocab`**
    - `ja_30K.vocab.symbolRemoved`
    - `en_20K.vocab.symbolRemoved`
    - `code_10K.vocab.symbolRemoved`
        
- マージする際，重複している語彙は削除される．
    - **`specialTokens.vocab`は最初に指定すること．**
    - 再推定したスコアを後で処理するときに，役物が最初に列挙されている必要があります．
```
$ python mergeVocab.py specialTokens.vocab ja_30K.vocab.symbolRemoved en_20K.vocab.symbolRemoved code_10K.vocab.symbolRemoved > ja30K_en20K_code10K.merged.vocab
```

- `specialTokens.vocab`の例：`/models/ver2/specialTokens.vocab`
    ```
    <unk>   0.0
    <s>     0.0
    </s>    0.0
    <mask>  0.0
    <pad>   0.0
    <CLS>   0.0
    <SEP>   0.0
    <EOD>   0.0
    ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁        0.0
    ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁ 0.0
    ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁  0.0
    ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁   0.0
    ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁    0.0
    ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁     0.0
    ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁      0.0
    ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁       0.0
    ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁        0.0
    ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁ 0.0
    ▁▁▁▁▁▁▁▁▁▁▁▁▁▁  0.0
    ▁▁▁▁▁▁▁▁▁▁▁▁▁   0.0
    ▁▁▁▁▁▁▁▁▁▁▁▁    0.0
    ▁▁▁▁▁▁▁▁▁▁▁     0.0
    ▁▁▁▁▁▁▁▁▁▁      0.0
    ▁▁▁▁▁▁▁▁▁       0.0
    ▁▁▁▁▁▁▁▁        0.0
    ▁▁▁▁▁▁▁ 0.0
    ▁▁▁▁▁▁  0.0
    ▁▁▁▁▁   0.0
    ▁▁▁▁    0.0
    ▁▁▁     0.0
    ▁▁      0.0
    ▁       0.0

            0.0
    ```

## 手作業でのトークン選定
- 必要であれば，この段階で `ja30K_en20K_code10K.merged.vocab` を目視で確認し，不必要なトークンを削除したり，必要なトークンを追加したりできる．
    - トークンスコアは次ステップで再推定するため，`0.0`などの適当な値で良い．

## 再推定
- pythonで書いているので時間がかかる（2GBのデータで2.5時間程度）
    - 将来的には，SentencePieceが持つtrainメソッドをそのまま使えるようにしたい．
- 再推定用にデータを作成
    - 前処理を行っていない日本語・英語・コードコーパスを連結
    ```
    $ cat ja.txt en.txt code.txt > merged.txt
    ```
- 再推定
    ```
    $ python reestimateScore.py \
        --vocab ja30K_en20K_code10K.merged.vocab \
        --data marged.txt \
        --output ja30K_en20K_code10K.merged.vocab.reestimated \
        --trainingMode EM \
        --maxEpoch 2
    ```

- 並列処理が可能な環境であれば`--trainingMode EMMT`とすることで，ある程度高速にEMアルゴリズムの学習が可能です．

## 語彙の後処理
- Special tokenやbyte tokenのスコアを`0.0`に直すために，以下の後処理スクリプトを走らせます．
    - `--numSpecialTokens`に指定した数だけ，先頭からスコアを`0.0`に置き換える単純な処理．
        - 例えば`/models/ver2/specialTokens.vocab`の役物＋バイトトークンを併せると289個．
            | 種類          | トークン数 |
            | ---           | --- |
            | <役物>        | 8 |
            | スペース連続  | 24 |
            | 改行          | 1 |
            | バイトトークン | 256 |
            | 合計          | 289 |
    - スペースを含むトークンのスコアを`-1.0`にする場合は`--spaceNegativeOne`オプションを使う．
        - スコアが-1.0で良いのかは要検討．
    ```
    $ postproc_vocab.py -v ja30K_en20K_code10K.merged.vocab.reestimated --numSpecialTokens 289 > ja30K_en20K_code10K.merged.vocab.reestimated.postproc
    ```

## VocabのModel化
- 完成モデルは空白スペースを取り扱いたいので，ベースになるSentencePieceモデルのダミーとして`split_by_whitespace=False`の設定を持つモデルを作っておく．
    - TODO: ダミーを作るとかいう回りくどいことはせずに，`vocab2model.py`内で直接設定を変更できるようにしたい．
    ```
    spm.SentencePieceTrainer.train(
        input='/path/to/some/data',
        model_prefix=dummy,
        vocab_size=30000,
        num_threads=72,
        train_extremely_large_corpus=True,
        normalization_rule_name='identity',
        user_defined_symbols='\n'
        max_sentencepiece_length=16,
        split_digits=True,
        byte_fallback=True,
        split_by_whitespace=False, # ここがモデル作成時と異なる
        allow_whitespace_only_pieces=True, 
        remove_extra_whitespaces=False, 
    )
    ```

- 同梱の `createModel.py` を使う場合は，例えば以下のようなコマンドでダミーを作成する．
データや語彙の規模などは何でも良い．

    ```
    python /home/tathi/work/tokenizer/createModel.py \
        --data /path/to/some/data \
        --vocabSize 30000 \
        --maxLength 16 \
        --prefix dummy
    ```
- vocabをmodelにする．
    ```
    $ python vocab2model.py \
        --vocab ja30K_en20K_code10K.merged.vocab.reestimated.postproc \
        --output ja30K_en20K_code10K.model \
        --baseModel dummy.model
        --unk-token <unk>
        --special-tokens <s> </s> <mask> <pad> <CLS> <EOD> <SEP>
    ```
