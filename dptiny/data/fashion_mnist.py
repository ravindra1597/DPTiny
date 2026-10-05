"""Fashion-MNIST dataset loader."""

import os
from typing import Optional, Tuple
import numpy as np
import gzip
from dptiny.backend import xp

# for downloading the four files
import urllib.request 

_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "dptiny", "fashion_mnist")

# The classes for our new fashion_mnist are: 
CLASSES = (
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
)

# I am telling my loader where the four Fashion-MNIST files are.
_BASE_URL = (
    "https://raw.githubusercontent.com/"
    "zalandoresearch/fashion-mnist/master/data/fashion"
)

_FILES = (
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
)

# The assignment clearly said :
# 'Reuse the cache afterwards, and never leave a half-written file 
# in it if a download is interrupted.'

# Now, I am creating the download helper
def download_file(url: str, path: str) -> None:
    # Reuseing an existing completed download.
    if os.path.exists(path):
        return

    temp_path = path + ".part"

    # Removing any partial file from an earlier interrupted download.
    if os.path.exists(temp_path):
        os.remove(temp_path)

    try:
        with urllib.request.urlopen(url) as response:
            with open(temp_path, "wb") as f:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)

        # Renaming only after the entire download succeeds.
        os.replace(temp_path, path)

    except Exception:
        # I am making sure to never leave a partial downloaded file behind.
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise

# Here we will download all the four files and cache them as well.
def download_fashion_mnist(data_home: Optional[str] = None) -> str:
    cache_dir = data_home if data_home is not None else _CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)

    for filename in _FILES:
        url = f"{_BASE_URL}/{filename}"
        path = os.path.join(cache_dir, filename)
        download_file(url, path)

    return cache_dir

# Reading a gzip compressed uint8 IDX file.
def read_idx(path: str) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        data = f.read()

    # An IDX file must contain at least the 4-byte basic header.
    if len(data) < 4:
        raise ValueError(f"Invalid IDX file: {path}")

    # First two bytes must be zero.
    if data[0] != 0 or data[1] != 0:
        raise ValueError(f"Invalid IDX header: {path}")

    # Third byte is the IDX data type.
    # 0x08 means unsigned byte (uint8).
    if data[2] != 0x08:
        raise ValueError(f"IDX file is not uint8: {path}")

    # Fourth byte  will tell us how many dimensions the array has.
    num_dimensions = data[3]

    if num_dimensions == 0:
        raise ValueError(f"Invalid IDX dimensions: {path}")

    header_size = 4 + 4 * num_dimensions

    if len(data) < header_size:
        raise ValueError(f"Incomplete IDX header: {path}")

    # Each dimension size is stored as a 4-byte big-endian integer.
    dimensions = []

    offset = 4

    for _ in range(num_dimensions):
        dimension = int.from_bytes(
            data[offset:offset + 4],
            byteorder="big",
        )
        dimensions.append(dimension)
        offset += 4

    expected_size = int(np.prod(dimensions))

    actual_size = len(data) - header_size

    if actual_size != expected_size:
        raise ValueError(
            f"IDX file size does not match header: {path}"
        )

    array = np.frombuffer(
        data,
        dtype=np.uint8,
        offset=header_size,
    )

    return array.reshape(tuple(dimensions))

# Ok, Now that we have the required fuctions ready, we will create the data loader. 
# I am taking the strucutre from the code of mnist.py and editing it here. 
def get_fashion_mnist(
    normalize: bool = True,
    flatten: bool = True,
    data_home: Optional[str] = None,
) -> Tuple["xp.ndarray", "xp.ndarray", "xp.ndarray", "xp.ndarray"]:
    """Load the Fashion-MNIST dataset.

    Args:
        normalize: If True, scale pixel values to [0, 1].
        flatten: If True, return images as (N, 784) vectors.
        data_home: Directory used to cache the downloaded files.

    Returns:
        (X_train, X_test, y_train, y_test) arrays.
    """

    cache_dir = download_fashion_mnist(data_home)
    # we add the training images to the training path variable.
    train_images_path = os.path.join(
        cache_dir, "train-images-idx3-ubyte.gz"
    )
    train_labels_path = os.path.join(
        cache_dir, "train-labels-idx1-ubyte.gz"
    )
    test_images_path = os.path.join(
        cache_dir, "t10k-images-idx3-ubyte.gz"
    )
    test_labels_path = os.path.join(
        cache_dir, "t10k-labels-idx1-ubyte.gz"
    )

    X_train = read_idx(train_images_path)
    y_train = read_idx(train_labels_path)
    X_test = read_idx(test_images_path)
    y_test = read_idx(test_labels_path)

    # Match the dtypes used by get_mnist().
    X_train = xp.asarray(X_train, dtype=xp.float32)
    X_test = xp.asarray(X_test, dtype=xp.float32)
    y_train = xp.asarray(y_train, dtype=xp.int32)
    y_test = xp.asarray(y_test, dtype=xp.int32)

    if normalize:
        X_train = X_train / 255.0
        X_test = X_test / 255.0

    if flatten:
        X_train = X_train.reshape(-1, 784)
        X_test = X_test.reshape(-1, 784)
    else:
        X_train = X_train.reshape(-1, 1, 28, 28)
        X_test = X_test.reshape(-1, 1, 28, 28)

    return X_train, X_test, y_train, y_test