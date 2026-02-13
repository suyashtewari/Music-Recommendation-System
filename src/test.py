import h5py
import os

BASE_DIR = os.path.dirname(__file__)
file_name = "TRAAAAW128F429D538.h5"
dataset_folder = os.path.join(BASE_DIR, "../data/small_dataset")
path = os.path.abspath(os.path.join(dataset_folder, file_name))

print("Trying to open:", path)
print("Exists?", os.path.exists(path))

with h5py.File(path, "r") as f:
    print(f.keys())
    print("METADATA: ", list(f["metadata"]["songs"]["title"].dtype.names))
    print("ANALYSIS: ", list(f["analysis"]["songs"].dtype.names))