import pandas as pd
from pathlib import Path

DATA_PATH = Path("data/processed/transactions_clean.parquet")
OUT_PATH = Path("data/processed/item_catalog.parquet")


def build_catalog(df: pd.DataFrame):

    catalog = (
        df.groupby("item_id")
        .agg(
            description=("description", lambda x: x.value_counts().idxmax()),
            avg_price=("price", "mean"),
            country_main=("country", lambda x: x.value_counts().idxmax()),
            nb_sales=("invoice", "nunique"),
            total_revenue=("amount", "sum")
        )
        .reset_index()
    )

    return catalog


def main():

    df = pd.read_parquet(DATA_PATH)

    catalog = build_catalog(df)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    catalog.to_parquet(OUT_PATH, index=False)

    print("Catalogue créé :", catalog.shape)
    print(catalog.head())


if __name__ == "__main__":
    main()
