from pathlib import Path
import duckdb

# Chemin vers le parquet nettoyé
path = Path("data/processed/transactions_clean.parquet")

def run():
    con = duckdb.connect(database=":memory:")

    # On lit directement le parquet 
    con.execute(f"""
        CREATE VIEW tx as
        SELECT * FROM read_parquet('{path.as_posix()}');
    """)

    print("\n===== RAPPORT SQL (DuckDB) =====\n")

    # Volume global
    print("Volume global :")
    print(con.execute("""
        SELECT
            COUNT(*) as nb_transactions,
            COUNT(DISTINCT user_id) as nb_clients,
            COUNT(DISTINCT item_id) as nb_produits
        FROM tx;
    """).fetchdf())

    # Période couverte
    print("\nPériode couverte :")
    print(con.execute("""
        SELECT
            MIN(ts) as debut,
            MAX(ts) as fin
        FROM tx;
    """).fetchdf())

    # Montants (stats principales)
    print("\nMontants (stats) :")
    print(con.execute("""
        SELECT
            MIN(amount) as min_amount,
            APPROX_QUANTILE(amount, 0.25) as q25,
            APPROX_QUANTILE(amount, 0.50) as mediane,
            APPROX_QUANTILE(amount, 0.75) as q75,
            MAX(amount) as max_amount,
            AVG(amount) as moyenne
        FROM tx;
    """).fetchdf())

    # Top pays
    print("\nTop 10 pays :")
    print(con.execute("""
        SELECT country, COUNT(*) as nb_transactions
        FROM tx
        GROUP BY country
        ORDER BY nb_transactions DESC
        LIMIT 10;
    """).fetchdf())

    # Top clients par CA
    print("\nTop 5 clients par chiffre d'affaires :")
    print(con.execute("""
        SELECT user_id, SUM(amount) as ca
        FROM tx
        GROUP BY user_id
        ORDER BY ca DESC
        LIMIT 5;
    """).fetchdf())

    # Top produits par CA
    print("\nTop 5 produits par chiffre d'affaires :")
    print(con.execute("""
        SELECT item_id, SUM(amount) as ca
        FROM tx
        GROUP BY item_id
        ORDER BY ca DESC
        LIMIT 5;
    """).fetchdf())

if __name__ == "__main__":
    run()
