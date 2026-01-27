# 📊 CRM & Recommendation System

Projet personnel de data analytics et machine learning simulant un système de connaissance client, segmentation et recommandation, inspiré de cas d’usage e-commerce / CRM.

---

## 🎯 Objectifs

- Analyser les comportements clients
- Segmenter la base via RFM
- Construire un moteur de recommandation
- Intégrer des critères business (CA, marge)
- Proposer un outil d’aide à la décision marketing

---

## 🏗️ Architecture du projet

├── data/  
│ ├── raw/ # Données brutes  
│ ├── processed/ # Données nettoyées et enrichies  
│ └── models/ # Modèles entraînés  
├── src/  
│ ├── data/ # Préparation / nettoyage  
│ ├── features/ # Segmentation RFM  
│ ├── models/ # Recommandation  
│ ├── evaluation/ # Évaluation offline  
│ ├── activation/ # Stratégie CRM  
│ └── app/ # Application Streamlit  
├── reports/ # Résultats & exports  
└── README.md  


---

## 🔄 Pipeline Data

1. Nettoyage des transactions
2. Construction des interactions user-item
3. Analyse SQL (DuckDB)
4. Segmentation RFM
5. Génération du catalogue produit
6. Entraînement du modèle ALS
7. Évaluation offline
8. Déploiement Streamlit

---

## 📈 Segmentation CRM (RFM)

Les clients sont segmentés selon :

- Recency : récence d’achat
- Frequency : fréquence
- Monetary : valeur

Segments :

- VIP
- Fidèles
- Occasionnels
- Dormants
- Nouveaux

Chaque segment est associé à une stratégie d’activation.

---

## 🤖 Moteur de recommandation

- Algorithme : ALS (Alternating Least Squares) implicite
- Données : historique d’achats
- Représentation : factorisation user-item
- Couche business : re-ranking par marge simulée

---

## 📊 Évaluation

Évaluation offline avec split temporel :

- Precision ≈ 0.15
- Recall ≈ 0.07
- MAP ≈ 0.09

Ces scores sont significatifs dans un contexte de recommandation à large catalogue (plusieurs milliers de produits), où la prédiction est complexe et fortement bruitée. Ils indiquent une performance nettement supérieure à un modèle aléatoire et confirment la capacité du système à capter des préférences utilisateurs pertinentes.


---

## 🖥️ Application Streamlit

Une application interactive qui permet de :

- Sélectionner un client
- Visualiser son segment
- Générer des recommandations
- Comparer appétence vs optimisation business

Pour lancer l'application, rien de plus simple, il suffit d'écrire dans le terminal : src/app/streamlit_app.py

---

## 🛠️ Technologies

- Python
- Pandas / NumPy
- DuckDB (SQL analytics)
- Implicit (ALS)
- Scipy (sparse matrix)
- Streamlit

---

## 🚀 Installation

python -m venv .venv  
source .venv/bin/activate  
pip install -r requirements.txt  


---

## ▶️ Exécution principale
Préparation des données :  
python src/data/clean_data.py  
Segmentation :  
python src/features/rfm_scoring_sql.py  
Entraînement :  
python src/models/train_cf.py  
Évaluation :  
python src/evaluation/evaluate_cf.py  

---

## 📌 Auteur
Mathieu Nobre                                   
                                   
Projet réalisé dans le cadre d’une démarche de montée en compétences en data science appliquée au marketing et à la relation client.    