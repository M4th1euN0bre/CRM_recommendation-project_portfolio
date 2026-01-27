import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import coo_matrix
from implicit.als import AlternatingLeastSquares

INTER_PATH = Path("data/processed/interactions.parquet")
RFM_PATH = Path("data/processed/rfm_scored.parquet")
MODEL_PATH = Path("data/models/als_model.npz")
CATALOG_PATH = Path("data/processed/item_catalog.parquet")


def build_matrix(df: pd.DataFrame):
    """Construit la matrice user-item sparse et les mappings"""
    users = df["user_id"].astype(str).unique()
    items = df["item_id"].astype(str).unique()

    user_to_idx = {u: i for i, u in enumerate(users)}
    idx_to_user = {i: u for u, i in user_to_idx.items()}

    item_to_idx = {it: i for i, it in enumerate(items)}
    idx_to_item = {i: it for it, i in item_to_idx.items()}

    rows = df["user_id"].astype(str).map(user_to_idx).to_numpy()
    cols = df["item_id"].astype(str).map(item_to_idx).to_numpy()
    values = df["total_amount"].to_numpy().astype(np.float32)

    mat = coo_matrix((values, (rows, cols)), shape=(len(users), len(items))).tocsr()
    return mat, user_to_idx, idx_to_user, item_to_idx, idx_to_item


def load_model(n_users: int, n_items: int):
    """Recharge le modèle ALS à partir des facteurs sauvegardés"""
    z = np.load(MODEL_PATH)
    model = AlternatingLeastSquares(factors=z["user_factors"].shape[1])
    # On injecte les facteurs
    model.user_factors = z["user_factors"]
    model.item_factors = z["item_factors"]

    # Petite sécurité
    assert model.user_factors.shape[0] == n_users
    assert model.item_factors.shape[0] == n_items

    return model


def get_user_segment(rfm: pd.DataFrame, user_id: str) -> str:
    row = rfm[rfm["user_id"] == user_id]
    if row.empty:
        return "Inconnu"
    return row.iloc[0]["segment"]


def simulate_item_margin(items: list) -> dict:
    """
    On simule une marge par item (dans un vrai cas, ça vient du référentiel produit).
    Ici : marge entre 10% et 40%.
    """
    rng = np.random.default_rng(42)
    margins = rng.uniform(0.10, 0.40, size=len(items))
    return {it: float(m) for it, m in zip(items, margins)}


def rerank_by_margin(recs: list, margin_map: dict, k: int = 10) -> list:
    """Reclasse les recommandations en privilégiant la marge"""
    scored = [(it, margin_map.get(it, 0.0)) for it in recs]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [it for it, _ in scored[:k]]


def main():
    # 1) Chargement données
    inter = pd.read_parquet(INTER_PATH)
    rfm = pd.read_parquet(RFM_PATH)
    catalog = pd.read_parquet(CATALOG_PATH)
    item_info = catalog.set_index("item_id").to_dict("index")



    # 2) Matrice + mappings
    mat, user_to_idx, idx_to_user, item_to_idx, idx_to_item = build_matrix(inter)

    # 3) Modèle
    model = load_model(n_users=mat.shape[0], n_items=mat.shape[1])

    # 4) Choix user (on prend un user au hasard parmi les VIP pour une démo sympa)
    vip_users = rfm[rfm["segment"] == "VIP"]["user_id"].astype(str).tolist()
    user_id = vip_users[0] if vip_users else inter["user_id"].astype(str).iloc[0]

    print("\n===== DÉMO RECOMMANDATION =====")
    print("Utilisateur :", user_id)
    print("Segment :", get_user_segment(rfm, user_id))

    # 5) Recos ALS
    uidx = user_to_idx[user_id]
    recs_idx, scores = model.recommend(uidx, mat[uidx], N=30)
    recs = [idx_to_item[i] for i in recs_idx]

    print("\nTop 10 recos (ALS brut) :")
    print(recs[:10])

    # 6) Exclure items déjà achetés (dans les interactions)
    already = set(inter[inter["user_id"].astype(str) == user_id]["item_id"].astype(str).tolist())
    recs_filtered = [it for it in recs if it not in already]

    print("\nTop 10 recos (filtrées achats passés) :")
    print(recs_filtered[:10])

    # 7) Re-ranking business (marge simulée)
    margin_map = simulate_item_margin(recs_filtered[:50])
    recs_reranked = rerank_by_margin(recs_filtered[:50], margin_map, k=10)

    print("\nTop 10 recos enrichies :\n")

    for it in recs_reranked:

        info = item_info.get(it, {})

        name = info.get("description", "N/A")
        price = info.get("avg_price", None)
        ca = info.get("total_revenue", None)

        print(f"- {it} | {name} | Prix moyen: {price:.2f}€ | CA total: {ca:.2f}€")


    # 8) Sauvegarde des recommandations pour réutilisation (dashboard / app)
    rows = []

    for it in recs_reranked:

        info = item_info.get(it, {})

        rows.append({
            "user_id": user_id,
            "segment": get_user_segment(rfm, user_id),
            "item_id": it,
            "description": info.get("description"),
            "avg_price": info.get("avg_price"),
            "total_revenue": info.get("total_revenue")
        })

    out = pd.DataFrame(rows)
    out_path = Path("reports/sample_recommendations.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print("\nSauvegarde :", out_path)


if __name__ == "__main__":
    main()
