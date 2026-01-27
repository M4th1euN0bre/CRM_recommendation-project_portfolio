import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import coo_matrix
from implicit.als import AlternatingLeastSquares


DATA_PATH = Path("data/processed/interactions.parquet")


def build_matrix(df, user_map, item_map):
    rows = df["user_id"].map(user_map).to_numpy()
    cols = df["item_id"].map(item_map).to_numpy()
    vals = df["total_amount"].to_numpy().astype(np.float32)

    return coo_matrix(
        (vals, (rows, cols)),
        shape=(len(user_map), len(item_map))
    ).tocsr()


def precision_recall_at_k(recommended, relevant, k=10):

    recommended = recommended[:k]

    rel = set(relevant)

    tp = len([x for x in recommended if x in rel])

    precision = tp / k if k else 0
    recall = tp / len(rel) if rel else 0

    return precision, recall


def average_precision(recommended, relevant, k=10):

    rel = set(relevant)
    score = 0.0
    hits = 0

    for i, item in enumerate(recommended[:k]):
        if item in rel:
            hits += 1
            score += hits / (i + 1)

    return score / min(len(rel), k) if rel else 0


def main():

    print("Chargement interactions...")
    df = pd.read_parquet(DATA_PATH)

    # Split temporel
    split_date = df["last_ts"].quantile(0.8)

    train = df[df["last_ts"] <= split_date]
    test = df[df["last_ts"] > split_date]

    print("Train size :", len(train))
    print("Test size :", len(test))

    # Users/items communs
    users = train["user_id"].unique()
    items = train["item_id"].unique()

    train = train[
        train["user_id"].isin(users) &
        train["item_id"].isin(items)
    ]

    test = test[
        test["user_id"].isin(users) &
        test["item_id"].isin(items)
    ]

    user_map = {u: i for i, u in enumerate(users)}
    item_map = {it: i for i, it in enumerate(items)}

    # Matrice train
    mat = build_matrix(train, user_map, item_map)

    print("Matrice train :", mat.shape)

    # Entraînement
    model = AlternatingLeastSquares(
        factors=64,
        regularization=0.05,
        iterations=20,
        random_state=42
    )
    model.fit(mat)

    # Évaluation
    print("\n===== ÉVALUATION OFFLINE =====\n")

    precisions = []
    recalls = []
    maps = []

    grouped_test = test.groupby("user_id")

    for user_id, group in grouped_test:

        if user_id not in user_map:
            continue

        uidx = user_map[user_id]

        relevant = group["item_id"].tolist()

        rec_idx, _ = model.recommend(uidx, mat[uidx], N=20)
        recs = [list(item_map.keys())[i] for i in rec_idx]

        p, r = precision_recall_at_k(recs, relevant, k=10)
        ap = average_precision(recs, relevant, k=10)

        precisions.append(p)
        recalls.append(r)
        maps.append(ap)

    print("Precision :", np.mean(precisions))
    print("Recall    :", np.mean(recalls))
    print("MAP       :", np.mean(maps))


if __name__ == "__main__":
    main()
