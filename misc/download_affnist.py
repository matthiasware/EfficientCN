import argparse
import os
import sys
import zipfile
from six.moves import urllib
from pathlib import Path

# affnist dataset
HOMEPAGE="https://www.cs.toronto.edu/~tijmen/affNIST/"
AFFNIST_CENTERED_TRAIN_URL = HOMEPAGE + "32x/just_centered/test.mat.zip"
AFFNIST_CENTERED_TEST_URL = HOMEPAGE + "32x/just_centered/training_and_validation.mat.zip"
AFFNIST_TRANSFORMED_TRAIN_URL = HOMEPAGE + "32x/transformed/training_and_validation_batches.zip"
AFFNIST_TRANSFORMED_TEST_URL = HOMEPAGE + "32x/transformed/test_batches.zip"


def download_and_unzip(url, p_data, force):
    print("#" * 100)
    print("Downloading {}".format(url))
    print("#" * 100)

    p_data_dir = p_data / url.split('/')[-2]
    filename = url.split("/")[-1]
    p_file_zip = p_data_dir / filename
    p_file = p_data_dir / p_file_zip.name[:-4]

    # DOWNLOAD
    if p_file.exists() and not force:
        print("File already downloaded: {}".format(p_file_zip))
        print("Use parameter --force to download again!")
        return
    p_data_dir.mkdir(exist_ok=True, parents=True)
    print("Saving to:   {}".format(p_file_zip))

    def progress_func(count, block_size, total_size):
        print("\rProgress     {:.0f}%".format(float(count * block_size) / float(total_size) * 100.), flush=True, end="")

    _, _ = urllib.request.urlretrieve(url, p_file_zip, progress_func)
    assert p_file_zip.exists()
    print( " Sucess")

    # UNZIP
    with zipfile.ZipFile(p_file_zip, 'r') as fd:
        print('Unzipping   ', p_file_zip, end="")
        fd.extractall(p_data_dir)
        fd.close()
        assert p_file.exists()
        print(' Success')

    # DELETE ZIP
    if p_file_zip.exists():
        p_file_zip.unlink()

def start_download(p_data, force):
    download_and_unzip(AFFNIST_CENTERED_TRAIN_URL, p_data, force)
    download_and_unzip(AFFNIST_CENTERED_TEST_URL, p_data, force)
    download_and_unzip(AFFNIST_TRANSFORMED_TRAIN_URL, p_data, force)
    download_and_unzip(AFFNIST_TRANSFORMED_TEST_URL, p_data, force)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Script for automatically downloading datasets')
    parser.add_argument("--save_to", default="./data/affnist")
    parser.add_argument("--force", default=False, type=bool)
    args = parser.parse_args()

    p_data = Path(args.save_to)
    start_download(p_data, args.force)