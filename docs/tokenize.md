# Tokenize Training Dataset with original tokenizer

GPT-Fugaku では、デフォルトの GPT2BPETokenizer(BytePairEncoding)ではなく、SentencePiece を用いた Tokenizer を使用しています。また、Tokenizer の version も複数あり、それぞれの Tokenizer で binarize した`.bin`, `.idx`ファイルを作成する必要があります。

## Tokenize するための環境

DeepSpeedFugaku のコードは、Training を行うための CPU 移植は行われていますが、Tokenize するための処理で必要なプロセスの CPU 移植は行えていません。また、Text Generation についても同様です。

そのため、Tokenize を行うためには、GPU 環境にて環境を構築し、Tokenize を行う必要があります。

### 環境構築

環境構築は通常の Megatron-DeepSpeed の環境構築方法と同じですが、`nltk`, `sentencepiece`などの学習においては使用しないライブラリをインストールする必要があります。

(`scripts/tokenize-gpu-requirements.txt`に CUDA11.7 環境でのセットアップ用の requirements.txt を用意しています。)
Megatron-DeepSpeed の環境構築方法の詳細については、[こちらの記事をご覧ください](https://zenn.dev/turing_motors/articles/04c1328bf6095a)

### dataset の用意

Fugaku で dataset を download しても良いですが、処理の用意さなどから、通常の CPU 環境での作業を推奨します.
(実は、Fugaku の login ノードは Fujitsu 製の chip ではないので、通常の pip install で作業ができますが、login ノードに負荷をかける行為になるので推奨できません)

(Fugaku 上に、以下の処理をした tokenize 前のデータと、一部の Tokenize 済みデータを`/data/hp190122/share/dataset`に置いてあります。適時使用してください。)

```bash

cd DeepSpeedFugaku

python -m venv .env

source .env/bin/activate

pip install wikiextractor
```

最新の Wikipeida dump をダウンロードします。
(https://github.com/rioyokotalab/DeepSpeedFugaku/blob/training/feature/v2-tokenizer/dataset/wikipedia/wikidump_download.py#L41 の`["ja]"`を`["ja", "en"]`とすると日本語と英語の wikipedia が download できます。)

```bash
python dataset/wikipedia/wikidump_download.py
```

実行すると`data/wikipedia/processed`に`ja`や`en`というディレクトリが生成されます。(`["ja"]`とした場合は、`ja`だけしか生成されません)

jsonl 形式の file にします。

`dataset/wikipedia/merge_files.sh`にある script を利用して、1 つにまとめる作業を行います。

Japanese Wikipedia:

```bash
bash dataset/wikipedia/merge_files.sh \
  data/wikipedia/processed/ja/AA \
  data/wikipedia/merged/ja \
  ja_merged
```

English Wikipedia:

```bash
bash dataset/wikipedia/merge_files.sh \
  data/wikipedia/processed/en/AA \
  data/wikipedia/merged/en \
  en_merged_1

bash dataset/wikipedia/merge_files.sh \
  data/wikipedia/processed/en/AB \
  data/wikipedia/merged/en \
  en_merged_2

bash dataset/wikipedia/merge_files.sh \
  data/wikipedia/merged/en \
  data/wikipedia/merged/en \
  en_merged
```

これで、Japanese Wiki, English Wiki のデータを用意できました。🎉

### Tokenize の実行

`DeepSpeedFugaku/.env`に仮想環境を作成している前提ですが、`scripts/`に Tokenizer-V1, Tokenizer-V2 それぞれの Tokenizer で data を binarize するための script を用意しています。(横田研 lab server にて動作するようにしていますが、適時それぞれの環境にあわせて変えてください。)

注意: `dataset/wikipedia/merged/ja/ja_merged.json`に Wikipedia のデータが用意されていることを前提にしています。
(事前に、`scripts/`にある shell script の内容を確認してから実行してください。)

```bash
# Tokenizer-V1
cd DeepSpeedFugaku

ybatch scripts/tokenize-v1.sh

```

https://github.com/rioyokotalab/DeepSpeedFugaku/blob/training/feature/v2-tokenizer/scripts/tokenize-v1.sh#L13-L14 の部分を変更することで、日本語と英語の vocab size を変更した Tokenizer での Tokenize を行うことができます。

```bash
# Tokenizer-V2

cd DeepSpeedFugaku

ybatch scripts/tokenize-v2.sh

```

binarized されたデータは`DeepSpeedFugaku/datasets/wikipedia/binarized`に出力されます。`.bin`と`.idx`のファイル両方があることを確認してください。

#### Hinaori(横田研 lab server)から Fugaku へのデータ Transfer

sftp を用いるのが良いでしょう。
横田研の lab server 上から fugaku へアクセスできるように ssh config と 秘密鍵を用意します。

その後

```bash
sftp fugaku

sftp > put -r <from> <to>
```

のような形で移動すれば良いでしょう。

例:

```bash
sftp fugaku
sftp> put -r datasets/wikipedia/binarized/v2-code20k_en40k_ja80k data/wikipedia/binarized
```
