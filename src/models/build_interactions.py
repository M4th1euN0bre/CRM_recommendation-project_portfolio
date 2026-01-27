import pandas as pd
from pathlib import Path

DATA_PATH = Path("data/processed/transactions_clean.parquet")
OUT_PATH = Path("data/processed/interactions.parquet")


def build_interactions(df: pd.DataFrame):

    # Agrégation user-item (montant total + quantité)
    inter = (
        df.groupby(["user_id", "item_id"])
        .agg(
            total_amount=("amount", "sum"),
            total_qty=("quantity", "sum"),
            nb_orders=("invoice", "nunique"),
            last_ts=("ts", "max")
        )
        .reset_index()
    )

    return inter


def main():

    df = pd.read_parquet(DATA_PATH)

    inter = build_interactions(df)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    inter.to_parquet(OUT_PATH, index=False)

    print("Interactions créées :", inter.shape)
    print(inter.head())


if __name__ == "__main__":
    main()
