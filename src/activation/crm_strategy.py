import pandas as pd
from pathlib import Path

RFM_path = Path("data/processed/rfm_scored.parquet")


def generate_strategies(df: pd.DataFrame):

    strategies = []

    for segment, count in df["segment"].value_counts().items():

        if segment == "VIP":
            strategies.append({
                "segment": segment,
                "taille": count,
                "objectif": "Fidélisation premium",
                "action": "Accès prioritaire, offres exclusives, bundles VIP",
                "kpi": "Taux de réachat / panier moyen"
            })

        elif segment == "Fidèles":
            strategies.append({
                "segment": segment,
                "taille": count,
                "objectif": "Montée en gamme",
                "action": "Cross-sell, recommandations personnalisées",
                "kpi": "Upsell rate"
            })

        elif segment == "Occasionnels":
            strategies.append({
                "segment": segment,
                "taille": count,
                "objectif": "Augmenter la fréquence",
                "action": "Offres limitées, reminders",
                "kpi": "Fréquence d'achat"
            })

        elif segment == "Dormants":
            strategies.append({
                "segment": segment,
                "taille": count,
                "objectif": "Réactivation",
                "action": "Campagne win-back, promotions ciblées",
                "kpi": "Taux de réactivation"
            })

        elif segment == "Nouveaux":
            strategies.append({
                "segment": segment,
                "taille": count,
                "objectif": "Onboarding",
                "action": "Welcome pack, recommandations initiales",
                "kpi": "Conversion 2e achat"
            })

    return pd.DataFrame(strategies)


def main():

    df = pd.read_parquet(RFM_path)

    plan = generate_strategies(df)

    print("\n===== STRATÉGIE CRM PAR SEGMENT =====\n")
    print(plan)

    out_path = Path("reports/crm_strategy.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plan.to_csv(out_path, index=False)

    print("\nPlan sauvegardé :", out_path)


if __name__ == "__main__":
    main()
