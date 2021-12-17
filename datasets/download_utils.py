import requests
import argparse
import os
import sys
import zipfile
from six.moves import urllib
from pathlib import Path


def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


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
        print("\rProgress     {:.0f}%".format(
            float(count * block_size) / float(total_size) * 100.), flush=True, end="")

    _, _ = urllib.request.urlretrieve(url, p_file_zip, progress_func)
    assert p_file_zip.exists()
    print(" Sucess")

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


def affnist_start_download(p_data, force):
    """
        Download from official fource.
        Note: this repo contains code to 
        generatre this dataset
    """
    # affnist dataset
    HOMEPAGE = "https://www.cs.toronto.edu/~tijmen/affNIST/"
    AFFNIST_CENTERED_TRAIN_URL = HOMEPAGE + "32x/just_centered/test.mat.zip"
    AFFNIST_CENTERED_TEST_URL = HOMEPAGE + \
        "32x/just_centered/training_and_validation.mat.zip"
    AFFNIST_TRANSFORMED_TRAIN_URL = HOMEPAGE + \
        "32x/transformed/training_and_validation_batches.zip"
    AFFNIST_TRANSFORMED_TEST_URL = HOMEPAGE + "32x/transformed/test_batches.zip"

    download_and_unzip(AFFNIST_CENTERED_TRAIN_URL, p_data, force)
    download_and_unzip(AFFNIST_CENTERED_TEST_URL, p_data, force)
    download_and_unzip(AFFNIST_TRANSFORMED_TRAIN_URL, p_data, force)
    download_and_unzip(AFFNIST_TRANSFORMED_TEST_URL, p_data, force)
