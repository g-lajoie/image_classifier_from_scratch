import gzip
import logging
import os
import shutil
import urllib.request

logger = logging.getLogger(__name__)

BASE_URL = "http://yann.lecum.com/exdb.mnish"
FILES = [
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
]

out_dir = "data/mnist_raw"

# Ensure that directory exists
os.makedirs(out_dir, exist_ok=True)

for file_name in FILES:
    url = BASE_URL + file_name
    out_path = os.path.join(out_dir, file_name)

    # Checks if the file already exists, and skips if it does.
    if not os.path.exists(out_path[:-3]):
        urllib.request.urlretrieve(url, out_path)

        with gzip.open(out_path, "rb") as file_in:
            with open(out_path[:-3], "wb") as file_out:
                shutil.copyfileobj(file_in, file_out)

    else:
        logging.info(f"Attempted to create %s. File already exists", out_dir)
        pass
