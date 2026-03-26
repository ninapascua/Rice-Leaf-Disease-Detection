# import os

# DATA_DIR = "data/raw"

# # create folder if it doesn't exist
# os.makedirs(DATA_DIR, exist_ok=True)

# print("Downloading rice leaf datasets...")

# # Kaggle datasets
# os.system("kaggle datasets download -d shayanriyaz/riceleafs -p data/raw --unzip")


# print("Download complete.")

import os
import subprocess
import sys

DATA_DIR = "data/raw"
os.makedirs(DATA_DIR, exist_ok=True)

print("Downloading rice leaf dataset...")

try:
    subprocess.run([
        sys.executable,
        "-m",
        "kaggle.cli",
        "datasets",
        "download",
        "-d",
        "loki4514/rice-leaf-diseases-detection",
        "-p",
        DATA_DIR,
        "--unzip"
    ], check=True)

    print("Download Complete.")

except subprocess.CalledProcessError:
    print("Download Failed.")