# Fugaku Training Checkpoint

## checkpoint

`/data/hp190122/share/fujii/checkpoints` に Fugaku での学習のチェックポイントを保存しています。

```
checkpoints
|-- ja-en-wiki
|-- ja-wiki
```

`ja-wiki` は日本語 wikipedia のみの学習、`ja-en-wiki` は日英 wikipedia の学習です。

checkpoint の`350m_dp512_v1_ja40K_en10K` は`<モデルサイズ>_<dp_size>_<tokenizer_version>_ja<ja_vocab_size>_en<en_vocab_size>` という命名規則に従っています。

## tokenizer

evaluation を行うためには、checkpoint と同じ tokenizer を使う必要があるかと思います。checkpoint の名前から、tokenizer を判別して、以下の tokenizer を利用してください。

### v1 tokenizer

https://github.com/rioyokotalab/DeepSpeedFugaku/tree/training/feature/v2-tokenizer/tokenizer/models/cc100ja1GB_cc100en1GB こちらに v1 tokenizer のモデルファイルがあります。

`training/feature/v2-tokenizer` branch であれば、`DeepSpeedFugaku/tokenizer/models/cc100ja1GB_cc100en1GB` にあります。 日本語と英語の vocab の比率から、適切な tokenizer を選択してください。

### v2 tokenizer

https://github.com/rioyokotalab/DeepSpeedFugaku/tree/training/feature/v2-tokenizer/tokenizer/models/ver2 こちらに v2 tokenizer のモデルファイルがあります。

v1 tokenizer のときと同様に、`training/feature/v2-tokenizer` branch であれば、`DeepSpeedFugaku/tokenizer/models/ver2` にあります。 日本語と英語, コードの vocab の比率から、適切な tokenizer を選択してください。

## further training

evaluation に使用するために、存在する checkpoint から更に学習を進めたい際は、[DeepSpeedFugaku](https://github.com/rioyokotalab/DeepSpeedFugaku)の最新の training branch から job script を探して、job を投げることができます。

基本的に、`experiments/scripts`に、学習に使用した job script が保存されています。

例えば、1.3b のモデルを v1 tokenizer (ja-10k, en-40k)で学習させた job script を探している場合は、`experiments/scripts/1.3b/JapaneseTokenizer/ja-wiki/1.3b_mp4_dp16-ja10K-en40K.sh`を参照してください。
