from pathlib import Path
import duckdb

path = Path("data/processed/transactions_clean.parquet")


def run():

    con = duckdb.connect(database=":memory:")

    # Charger les transactions
    con.execute(f"""
        CREATE VIEW tx as
        SELECT * FROM read_parquet('{path.as_posix()}');
    """)

    print("\n===== SEGMENTATION RFM =====\n")

    # Date de référence = dernière date + 1 jour
    ref_date = con.execute("""
        SELECT MAX(ts) + INTERVAL 1 DAY FROM tx;
    """).fetchone()[0]

    print("Date de référence :", ref_date)

    # Table RFM
    rfm = con.execute(f"""
        WITH table_intermediaire as (
            SELECT
                user_id,
                MAX(ts) as last_purchase,
                COUNT(DISTINCT invoice) as frequency,
                SUM(amount) as monetary
            FROM tx
            GROUP BY user_id
        ),

        rfm_calc as (
            SELECT
                user_id,
                DATE_DIFF('day', last_purchase, TIMESTAMP '{ref_date}') as recency,
                frequency,
                monetary
            FROM table_intermediaire
        )

        SELECT *
        FROM rfm_calc;
    """).fetchdf()

    print(rfm.head())

    print("\nRésumé RFM :")
    print(rfm.describe())

    # Sauvegarde
    out_path = Path("data/processed/rfm_table.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rfm.to_parquet(out_path, index=False)

    print("\nTable RFM sauvegardée :", out_path)


if __name__ == "__main__":
    run()
