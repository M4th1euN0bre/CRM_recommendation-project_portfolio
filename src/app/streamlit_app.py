import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.sparse import coo_matrix
from implicit.als import AlternatingLeastSquares


# ========================
# CHEMINS
# ========================

INTER_PATH = Path("data/processed/interactions.parquet")
RFM_PATH = Path("data/processed/rfm_scored.parquet")
CATALOG_PATH = Path("data/processed/item_catalog.parquet")
MODEL_PATH = Path("data/models/als_model.npz")


# ========================
# OUTILS
# ========================

@st.cache_data
def load_data():
    """Charge toutes les données"""
    inter = pd.read_parquet(INTER_PATH)
    rfm = pd.read_parquet(RFM_PATH)
    catalog = pd.read_parquet(CATALOG_PATH)
    return inter, rfm, catalog


@st.cache_resource
def load_model(inter):

    # Construction matrice
    users = inter["user_id"].astype(str).unique()
    items = inter["item_id"].astype(str).unique()

    user_to_idx = {u: i for i, u in enumerate(users)}
    idx_to_user = {i: u for u, i in user_to_idx.items()}

    item_to_idx = {it: i for i, it in enumerate(items)}
    idx_to_item = {i: it for it, i in item_to_idx.items()}

    rows = inter["user_id"].astype(str).map(user_to_idx).to_numpy()
    cols = inter["item_id"].astype(str).map(item_to_idx).to_numpy()
    vals = inter["total_amount"].to_numpy().astype(np.float32)

    mat = coo_matrix(
        (vals, (rows, cols)),
        shape=(len(users), len(items))
    ).tocsr()

    # Chargement facteurs
    z = np.load(MODEL_PATH)

    model = AlternatingLeastSquares(factors=z["user_factors"].shape[1])
    model.user_factors = z["user_factors"]
    model.item_factors = z["item_factors"]

    return model, mat, user_to_idx, idx_to_item


def get_segment(rfm, user_id):

    row = rfm[rfm["user_id"] == user_id]

    if row.empty:
        return "Inconnu"

    return row.iloc[0]["segment"]

def simulate_item_margin(item_ids):
    """Simule une marge par produit (stable pour la démo)"""
    rng = np.random.default_rng(42)
    margins = rng.uniform(0.10, 0.40, size=len(item_ids))
    return {it: float(m) for it, m in zip(item_ids, margins)}


def rerank_business(recs, scores, margin_map, k=10):
    """
    Re-ranking simple : score_final = score_ALS * marge
    recs: liste item_id
    scores: liste scores ALS correspondants
    """
    rows = []
    for it, s in zip(recs, scores):
        m = margin_map.get(it, 0.0)
        rows.append((it, float(s), m, float(s) * m))

    rows.sort(key=lambda x: x[3], reverse=True)
    return rows[:k]


# ========================
# APP
# ========================

def main():

    st.set_page_config(
        page_title="CRM Recommandation System",
        layout="wide"
    )

    st.title("📊 CRM & Recommendation System")
    st.markdown("Simulation d’un outil de personnalisation marketing")

    # Chargement
    inter, rfm, catalog = load_data()
    model, mat, user_map, item_map = load_model(inter)

    # Sélection utilisateur
    users = sorted(inter["user_id"].astype(str).unique())

    user_id = st.selectbox(
        "Sélectionner un client :",
        users
    )

    segment = get_segment(rfm, user_id)

    st.subheader("Profil client")
    st.write(f"**ID :** {user_id}")
    st.write(f"**Segment :** {segment}")

    mode = st.radio(
        "Mode de recommandation :",
        ["Appétence (ALS pur)", "Optimisation business (ALS × marge)"],
        horizontal=True
    )


    # Bouton reco
    if st.button("Générer recommandations"):

        uidx = user_map[user_id]
        
        rec_idx, scores = model.recommend(uidx, mat[uidx], N=50)
        recs = [item_map[i] for i in rec_idx]
        scores = list(scores)

        # Exclure déjà achetés
        already = set(
            inter[inter["user_id"].astype(str) == user_id]["item_id"].astype(str).tolist()
        )

        filtered = [(it, sc) for it, sc in zip(recs, scores) if it not in already]

        # On prend un pool pour rerank ensuite
        pool = filtered[:30]
        pool_items = [x[0] for x in pool]
        pool_scores = [x[1] for x in pool]

        item_info = catalog.set_index("item_id").to_dict("index")

        if mode == "Appétence (ALS pur)":
            final = [(it, sc, None, None) for it, sc in zip(pool_items[:10], pool_scores[:10])]
        else:
            margin_map = simulate_item_margin(pool_items)
            final = rerank_business(pool_items, pool_scores, margin_map, k=10)

        rows = []
        for it, sc, m, sf in final:
            info = item_info.get(it, {})
            rows.append({
                "Produit": info.get("description", it),
                "Score ALS (appétence)": round(float(sc), 4),
                "Marge simulée": None if m is None else round(float(m), 3),
                "Score final": None if sf is None else round(float(sf), 4),
                "Prix moyen (€)": round(float(info.get("avg_price", 0)), 2),
                "CA produit (€)": round(float(info.get("total_revenue", 0)), 2),
            })

        df_out = pd.DataFrame(rows)

        st.subheader("Recommandations personnalisées")
        st.dataframe(df_out, use_container_width=True)



if __name__ == "__main__":
    main()
