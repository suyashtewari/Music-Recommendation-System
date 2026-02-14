import os
from pathlib import Path
import shutil
import random

BASE_DIR = Path(__file__).resolve().parent.parent
SOURCE_DIR = BASE_DIR / "data" / "millionsongsubset"
TARGET_DIR = BASE_DIR / "data" / "small_dataset"
LIMIT = 1200

os.makedirs(TARGET_DIR, exist_ok=True)

all_files = []

for root, _, files in os.walk(SOURCE_DIR):
    for f in files:
        if f.endswith(".h5"):
            all_files.append(Path(root) / f)

random.shuffle(all_files)

for file in all_files[:LIMIT]:
    shutil.copy(file, TARGET_DIR)

print(f"Copied {LIMIT} songs.")
