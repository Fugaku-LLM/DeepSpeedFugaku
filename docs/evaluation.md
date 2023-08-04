# evaluation

## sample 抽出

実際に Training で用いられているデータセットから、ランダムにサンプルを抽出するためのスクリプトの使い方について解説します。

なお、[docs/tokenize.md](docs/tokenize.md) の dataset の用意の手順に従って、`merged.json`を用意していることを前提にしています。

Fugaku 以外の CPU 環境で実行する場合は`evaluation/scripts/extract.sh`を使用し、Fugaku 上で実行する場合は`evaluation/scripts/extract_fugaku.sh`を使用してください。

```bash
python evaluation/text_extractor.py \
  --input dataset/wikipedia/merged/ja/ja_merged.json \
  --output evaluation/out/samples.txt \
  --text-max-len $MAX_LENGTH \
  --num-samples $NUM_SAMPLES \
  --random-choice
```

基本的には、`--input`に`merged.json`を指定して、そこから sample を抽出します。

取り出したい sample のテキスト長についは`--text-max-len`にて指定可能です。また、抽出する sample 数に関しては`--num-samples`で指定できます。

注意: `--random-choice`をつけても、`evaluation/text_extractor.py`は file を先頭から読んでいる関係上、完全にデータ中からランダムに抽出されるわけではありません。完全なランダム性が必要な場合は、`for line in f:`ではなく`f.readlines()`を用いた別の実装が必要ですが、これを行うと、メモリを大量に消費しますし、実行に時間がかかります。
