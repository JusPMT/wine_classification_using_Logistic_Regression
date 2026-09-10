"""
Module thu thap va tai tap du lieu Wine Quality (Red Wine)
Nguon: UCI Machine Learning Repository / Cortez et al., 2009
"""

import os
import pandas as pd

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
RAW_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw", "winequality-red.csv")

def download_dataset(url: str = DATA_URL, destination: str = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Tai du lieu tu UCI Machine Learning Repository va luu vao thu muc data/raw.
    """
    destination = os.path.abspath(destination)
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    
    print(f"Dang tai du lieu tu: {url}")
    df = pd.read_csv(url, sep=";")
    df.to_csv(destination, index=False)
    print(f"Da luu du lieu goc thanh cong tai: {destination}")
    print(f"Kich thuoc tap du lieu: {df.shape[0]} mau, {df.shape[1]} cot.")
    return df

if __name__ == "__main__":
    download_dataset()
