import pandas as pd
from pathlib import Path

RAW_PATH = Path("data/raw/online_retail.csv")
OUT_PATH = Path("data/processed/transactions_clean.parquet")

def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    # Standardise le nom des colonnes
    df = df.rename(columns={
        "Invoice": "invoice",
        "StockCode": "item_id",
        "Description": "description",
        "Quantity": "quantity",
        "InvoiceDate": "ts",
        "Price": "price",
        "Customer ID": "customer_id",
        "Country": "country",
    })

    # Drop les NA
    df = df.dropna(subset=["customer_id", "item_id", "ts"])

    # conversion des id en string
    df["user_id"] = df["customer_id"].astype(int).astype(str)
    df["item_id"] = df["item_id"].astype(str)

    # Parse datetime
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    df = df.dropna(subset=["ts"])

    # on garde que les achats positifs
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["quantity", "price"])
    df = df[(df["quantity"] > 0) & (df["price"] > 0)]

    # montant
    df["amount"] = df["quantity"] * df["price"]

    # description
    df["description"] = df["description"].fillna("").astype(str).str.strip()

    # Select final columns
    out = df[[
        "invoice",
        "user_id",
        "item_id",
        "ts",
        "quantity",
        "price",
        "amount",
        "country",
        "description",
    ]].copy()

    return out

def main():
    print(f"Loading: {RAW_PATH}")
    df = pd.read_csv(RAW_PATH, encoding="ISO-8859-1")

    tx = clean_transactions(df)

    # Ensure output folder exists
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Save as parquet (fast + clean)
    tx.to_parquet(OUT_PATH, index=False)

    print("Saved:", OUT_PATH)
    print("Clean shape:", tx.shape)
    print(tx.head())

if __name__ == "__main__":
    main()
