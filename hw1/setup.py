import os
import urllib.request
import zipfile

GLOVE_URL = "https://nlp.stanford.edu/data/glove.6B.zip"
ZIP_NAME = "glove.6B.zip"


def _download_if_missing():
    if os.path.isfile(ZIP_NAME):
        return
    print(f"Downloading {GLOVE_URL}...")
    urllib.request.urlretrieve(GLOVE_URL, ZIP_NAME)


def _unzip_if_missing():
    if os.path.isfile("glove.6B.300d.txt"):
        return
    with zipfile.ZipFile(ZIP_NAME, "r") as zf:
        zf.extractall(".")


def main():
    _download_if_missing()
    _unzip_if_missing()


if __name__ == "__main__":
    main()
