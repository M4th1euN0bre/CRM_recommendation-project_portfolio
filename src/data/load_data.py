import pandas as pd
from pathlib import Path

path = Path("data/raw/online_retail.csv")

def load_data():
    df = pd.read_csv(path, encoding="ISO-8859-1")
    print("Shape:", df.shape)
    print(df.head())
    return df


if __name__ == "__main__":
    load_data()
