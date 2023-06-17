import os
from IPython import embed
from urllib.request import urlretrieve


class Config(object):
    def __init__(self, lang="ja"):
        self.corpus_name = "{}_wiki".format(lang)

        self.download_link = "https://dumps.wikimedia.org/{}wiki/latest/{}wiki-latest-pages-articles.xml.bz2".format(  # noqa
            lang, lang
        )
        self.raw_data_dir = "data/wikipedia/raw_data/{}".format(lang)
        os.makedirs(self.raw_data_dir, exist_ok=True)
        # self.raw_data_path = "{}/{}_wiki.json.gz".format(self.raw_data_dir, lang)
        self.raw_data_path = "{}/{}_xml.bz2".format(self.raw_data_dir, lang)

        # splitting into smaller files and convert them into loose json
        self.processed_data_dir = "data/wikipedia/processed/{}".format(lang)
        os.makedirs(self.processed_data_dir, exist_ok=True)
        self.chunk_size = "100M"


def download_data(config):
    if not os.path.exists(config.raw_data_path):
        print(f"Downloading {config.download_link} to {config.raw_data_path}")
        urlretrieve(config.download_link, config.raw_data_path)
        print(f"Successfully downloaded {config.raw_data_path}")


def preproces_data(config):
    input_file = config.raw_data_path
    output_dir = config.processed_data_dir
    chunk_size = config.chunk_size
    cmd = "python -m wikiextractor.WikiExtractor {} --o {} --bytes {} --json".format(
        input_file, output_dir, chunk_size
    )
    os.system(cmd)


if __name__ == "__main__":
    for lang in ["ja"]:
        config = Config(lang=lang)
        download_data(config)
        preproces_data(config)
