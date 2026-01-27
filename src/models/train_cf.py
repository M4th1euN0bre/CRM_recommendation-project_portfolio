import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import coo_matrix
from implicit.als import AlternatingLeastSquares


DATA_PATH = Path("data/processed/interactions.parquet")
MODEL_PATH = Path("data/models/als_model.npz")


def build_matrix(df: pd.DataFrame):
    """
    Construit la matrice user-item sparse
    """

    users = df["user_id"].astype(str).unique()
    items = df["item_id"].astype(str).unique()

    user_to_idx = {u: i for i, u in enumerate(users)}
    item_to_idx = {it: i for i, it in enumerate(items)}

    rows = df["user_id"].astype(str).map(user_to_idx).to_numpy()
    cols = df["item_id"].astype(str).map(item_to_idx).to_numpy()

    # Poids = montant total (signal business)
    values = df["total_amount"].to_numpy().astype(np.float32)

    mat = coo_matrix(
        (values, (rows, cols)),
        shape=(len(users), len(items))
    ).tocsr()

    return mat, user_to_idx, item_to_idx


def train_als(mat):

    model = AlternatingLeastSquares(
        factors=64,
        regularization=0.05,
        iterations=20,
        random_state=42
    )

    model.fit(mat)

    return model


def main():

    df = pd.read_parquet(DATA_PATH)
    mat, user_map, item_map = build_matrix(df)
    print("Taille matrice :", mat.shape)

    print("Entraînement ALS...")
    model = train_als(mat)

    # Sauvegarde
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        MODEL_PATH,
        user_factors=model.user_factors,
        item_factors=model.item_factors
    )

    print("Modèle sauvegardé :", MODEL_PATH)


if __name__ == "__main__":
    main()
