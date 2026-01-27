from pathlib import Path
import duckdb

path = Path("data/processed/transactions_clean.parquet")


def run():

    con = duckdb.connect(database=":memory:")

    # Charger transactions
    con.execute(f"""
        CREATE VIEW tx as
        SELECT * FROM read_parquet('{path.as_posix()}');
    """)

    print("\n===== SCORING RFM =====\n")

    # Date de référence
    ref_date = con.execute("""
        SELECT MAX(ts) + INTERVAL 1 DAY FROM tx;
    """).fetchone()[0]

    # Table RFM avec scores
    rfm = con.execute(f"""
        WITH base as (
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
            FROM base
        ),

        rfm_score as (
            SELECT
                *,
                NTILE(5) OVER (ORDER BY recency DESC) as r_score,
                NTILE(5) OVER (ORDER BY frequency) as f_score,
                NTILE(5) OVER (ORDER BY monetary) as m_score
            FROM rfm_calc
        )

        SELECT
            *,
            r_score || f_score || m_score as rfm_code,

            CASE
                WHEN r_score >= 4 AND f_score >= 4 AND m_score >= 4 THEN 'VIP'
                WHEN r_score >= 4 AND f_score >= 3 THEN 'Fidèles'
                WHEN r_score >= 4 AND f_score = 1 THEN 'Nouveaux'
                WHEN r_score <= 2 THEN 'Dormants'
                ELSE 'Occasionnels'
            END as segment

        FROM rfm_score;
    """).fetchdf()

    print(rfm.head())

    # Distribution des segments
    print("\nDistribution des segments :")
    print(rfm["segment"].value_counts())

    # Sauvegarde
    out_path = Path("data/processed/rfm_scored.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rfm.to_parquet(out_path, index=False)

    print("\nTable scorée sauvegardée :", out_path)


if __name__ == "__main__":
    run()
