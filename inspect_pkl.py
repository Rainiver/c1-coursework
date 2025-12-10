import pickle
import numpy as np

try:
    with open("coursework_dataset.pkl", "rb") as f:
        data = pickle.load(f)
        print("Keys:", data.keys())
        for k, v in data.items():
            if isinstance(v, np.ndarray):
                print(f"{k}: shape={v.shape}, dtype={v.dtype}")
            else:
                print(f"{k}: {type(v)}")
except Exception as e:
    print(f"Error: {e}")
