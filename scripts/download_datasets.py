import os

DATA_DIR = "data/raw"

# create folder if it doesn't exist
os.makedirs(DATA_DIR, exist_ok=True)

print("Downloading rice leaf datasets...")

# Kaggle datasets
os.system("kaggle datasets download -d shayanriyaz/riceleafs -p data/raw --unzip")
os.system("kaggle datasets download -d anshulm257/rice-disease-dataset -p data/raw --unzip")
os.system("kaggle datasets download -d shrupyag001/philippines-rice-diseases -p data/raw --unzip")

print("Download complete.")