import numpy as np
import pickle
from collections import namedtuple
import sys
import tarfile
import urllib.request
import os

CifarData = namedtuple("CifarData", ["data", "target"])


def download_cifar10(data_dir: str):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    url = "http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = url.split("/")[-1]
    filepath = os.path.join(data_dir, filename)
    if not os.path.exists(filepath):

        def _progress(count, block_size, total_size):
            sys.stdout.write(
                "\r>> Downloading %s %.1f%%"
                % (filename, float(count * block_size) / float(total_size) * 100.0)
            )
            sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(url, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print("Successfully downloaded", filename, statinfo.st_size, "bytes.")
    tarfile.open(filepath, "r:gz").extractall(os.path.dirname(data_dir))
    os.remove(filepath)


def unpickle(file: str) -> dict:
    with open(file, "rb") as fo:
        dict = pickle.load(fo, encoding="bytes")
    return dict


def load_cifar10(data_dir: str) -> CifarData:
    # download data if needed
    if not os.path.exists(data_dir):
        download_cifar10(data_dir)

    train_data = None
    train_labels = []

    for i in range(1, 6):
        data_dic = unpickle(data_dir + "/data_batch_{}".format(i))
        if i == 1:
            train_data = data_dic[b"data"]
        else:
            train_data = np.vstack((train_data, data_dic[b"data"]))
        train_labels += data_dic[b"labels"]

    return CifarData(data=train_data, target=train_labels)


def rgb_to_gray(rgb_image: np.ndarray):
    return np.uint8(
        0.299 * rgb_image[:1024]
        + 0.587 * rgb_image[1024:2048]
        + 0.114 * rgb_image[2048:]
    )


def load_cifar10_grayscale(data_dir: str) -> CifarData:
    if not os.path.exists(data_dir):
        download_cifar10(data_dir)

    train_data = None
    train_labels = []

    for i in range(1, 6):
        data_dic = unpickle(data_dir + "/data_batch_{}".format(i))
        if i == 1:
            train_data = data_dic[b"data"]
        else:
            train_data = np.vstack((train_data, data_dic[b"data"]))
        train_labels += data_dic[b"labels"]

    train_data = np.apply_along_axis(rgb_to_gray, axis=1, arr=train_data)

    return CifarData(data=train_data, target=train_labels)
