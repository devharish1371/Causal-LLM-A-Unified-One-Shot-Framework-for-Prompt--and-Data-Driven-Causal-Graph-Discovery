import numpy as np
import pandas as pd

def convert_npy_to_csv(npy_path, csv_path):
    try:
        data = np.load(npy_path)
        pd.DataFrame(data).to_csv(csv_path, index=False, header=False)
        print(f"Successfully converted {npy_path} to {csv_path} (no header).")
    except FileNotFoundError:
        print(f"Error: {npy_path} not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
