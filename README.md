# Projet ML Retail — Analyse Comportementale Clientèle

Projet de Machine Learning complet sur une base de données e-commerce 
de cadeaux (4 372 clients, 52 features).  
Objectif : segmenter les clients, prédire le churn et estimer 
le panier moyen par transaction.

---

## Résultats obtenus

| Tâche | Algorithme | Résultat |
|-------|-----------|---------|
| Segmentation | KMeans + PCA (K=3) | 3 segments — fallback automatique |
| Churn | RandomForest + Optuna (30 trials) | ROC-AUC = 0.8681 — Accuracy = 80% |
| Panier moyen | RandomForest + Optuna (30 trials) | MAE = 15.72 £ — R² = 0.802 |

---

## Description

Ce projet couvre la chaîne complète de traitement en data science :

- Exploration et visualisation des données (valeurs manquantes, 
  outliers, corrélations)
- Nettoyage et préparation des données (pipeline complet en 8 étapes)
- Clustering non supervisé (KMeans + PCA, K=3 sélectionné 
  automatiquement par silhouette score avec fallback)
- Classification supervisée (prédiction du churn — 
  RandomForest + Optuna, 30 trials)
- Régression supervisée (estimation du panier moyen — 
  RandomForest + Optuna, 30 trials)
- Déploiement d'une interface web avec Flask 
  ("Analyse Client en Temps Réel")

---

## Structure du projet
projet_ml_retail/
├── app/
│   ├── templates/
│   │   └── index.html              # Interface web (formulaire + résultats)
│   └── app.py                      # Application Flask
├── data/
│   ├── processed/
│   │   ├── customers_segmented.csv # Données avec labels de clusters
│   │   └── data_clean.csv          # Données nettoyées (4320 lignes, 48 col.)
│   ├── raw/
│   │   └── data_original.csv       # Données brutes (4372 lignes, 52 col.)
│   └── train_test/
│       ├── train.csv               # Données train (3456 lignes)
│       ├── test.csv                # Données test (864 lignes)
│       ├── X_train_norm.csv        # Features train normalisées
│       └── X_test_norm.csv         # Features test normalisées
├── models/
│   ├── kmeans.pkl                  # Modèle KMeans (K=3)
│   ├── pca.pkl                     # Modèle PCA (4 composantes, 96.5%)
│   ├── scaler_cluster.pkl          # Scaler clustering
│   ├── cluster_features.pkl        # Features utilisées pour le clustering
│   ├── churn_model.pkl             # Modèle RandomForest churn
│   ├── scaler_clf.pkl              # Scaler classification
│   ├── churn_columns.pkl           # Colonnes du modèle churn
│   ├── regression_model.pkl        # Modèle RandomForest panier moyen
│   ├── scaler_reg.pkl              # Scaler régression
│   └── reg_columns.pkl             # Colonnes du modèle régression
├── reports/
│   ├── model_metrics.json          # Métriques sauvegardées
│   ├── cluster_intervals_clean.csv # Intervalles min/max par cluster
│   ├── recommended_labels.txt      # Labels recommandés pour clusters
│   ├── pca_variance.png            # Variance expliquée par composante
│   ├── pca_2d.png                  # Projection PCA 2D
│   ├── clusters_2d.png             # Visualisation 2D des 3 clusters
│   ├── confusion_RandomForest_Churn.png  # Matrice de confusion
│   ├── roc_RandomForest_Churn.png        # Courbe ROC (AUC=0.87)
│   ├── feature_importance_churn.png      # Top 20 features importantes
│   ├── regression_reel_vs_predit.png     # Réel vs Prédit régression
│   ├── distribution_churn.png            # Distribution variable cible
│   ├── matrice_correlation.png           # Heatmap de corrélation
│   ├── boxplots_outliers.png             # Boxplots outliers
│   └── valeurs_manquantes.png            # Visualisation des NaN
├── src/
│   ├── cluster_labels.py           # Labels métier des 3 clusters
│   ├── preprocessing.py            # Pipeline complet (8 étapes)
│   ├── train_model.py              # Clustering + Classification + Régression
│   ├── predict.py                  # Prédiction pour un nouveau client
│   ├── test.py                     # Analyse détaillée des clusters
│   ├── debug_clustering.py         # Diagnostic du clustering
│   ├── diagnostic_complet.py       # Diagnostic complet des clusters
│   └── utils.py                    # Fonctions utilitaires réutilisables
├── .gitignore
├── README.md
└── requirements.txt

---

## Installation

### 1. Cloner le dépôt

```bash
git clone https://github.com/TON_USERNAME/projet_ml_retail.git
cd projet_ml_retail
```

### 2. Créer l'environnement virtuel

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

---

## Guide d'utilisation

Exécuter les scripts dans cet ordre depuis le dossier `src/` :

### Étape 1 — Preprocessing

```bash
cd src
python preprocessing.py
```

Ce que ça fait :
- Parse les dates (RegistrationDate → RegAnciennete) et les IPs 
  (LastLoginIP → IP_privee)
- Supprime **28 colonnes** inutiles ou à risque de data leakage 
  (CustomerID, ChurnRiskCategory, LoyaltyLevel, RFMSegment...)
- Corrige les valeurs aberrantes (SupportTicketsCount : -1 et 999 → NaN)
- Supprime les 52 lignes avec MonetaryTotal ≤ 0
- Feature engineering : AvgBasketValue, CancelRatio
- Encodage : ordinal (PreferredTimeOfDay) + one-hot encoding
- Split stratifié 80/20 (anti data leakage)
- Imputation post-split : Age (médiane=49), SupportTicketsCount (médiane=2)
- Normalisation post-split : StandardScaler sur 24 colonnes continues

Résultat : `data/processed/data_clean.csv` (4320 lignes × 48 colonnes)  
+ `data/train_test/train.csv` (3456 lignes) + `test.csv` (864 lignes)

### Étape 2 — Entraînement des modèles

```bash
python train_model.py
```

Ce que ça fait :
- Charge `train.csv` et `test.csv` séparément (pas de re-split)
- **Clustering** : KMeans + PCA (4 composantes, 96.5% variance), 
  K=3 (fallback automatique — les clients Ultra-VIP forment 
  un cluster isolé à 0.3%)
- **Classification churn** : RandomForest + Optuna (30 trials), 
  ROC-AUC=0.87, Accuracy=80%
- **Régression panier moyen** : RandomForest + Optuna (30 trials), 
  MAE=15.72£, R²=0.802
- Sauvegarde les modèles dans `models/`
- Génère les graphiques dans `reports/`

Durée : environ 3 à 5 minutes

### Étape 3 — Analyse des clusters

```bash
python test.py
```

Affiche les statistiques détaillées (moyenne, min, max, taux de churn) 
par cluster sur les données brutes non normalisées.  
Génère `reports/cluster_intervals_clean.csv` et 
`reports/recommended_labels.txt`.

### Étape 4 — Test de prédiction

```bash
python predict.py
```

Prédit le cluster, le risque de churn et le panier moyen estimé 
pour un client test défini dans le script.

### Étape 5 — Interface web Flask

```bash
cd ../app
python app.py
```

Ouvrir dans le navigateur : [http://127.0.0.1:5000](http://127.0.0.1:5000)

Renseigner les **9 champs** et cliquer sur **"Analyser ce client"** 
pour obtenir :
- Le segment client (Cluster 0, 1 ou 2) avec label métier
- Le statut churn (Stable / À surveiller / À risque)
- La probabilité de churn en %
- Le panier moyen estimé en £
- Des recommandations métier personnalisées

---

## Résultats des 3 clusters (données brutes)

| Cluster | Profil | Nb clients | % | Churn |
|---------|--------|-----------|---|-------|
| 0 | Clients faible activité (Freq. moy. 2, Monetary 627£) | 2 468 | 57.9% | 29.4% |
| 1 | Clients Ultra-VIP (Freq. moy. 105, Monetary 110 482£) | 13 | 0.3% | **53.8%** ⚠️ |
| 2 | Clients Premium réguliers (Freq. moy. 9, Monetary 2 986£) | 1 779 | 41.8% | 38.3% |

---

## Top 10 features — Prédiction du churn

| Feature | Importance |
|---------|-----------|
| CustomerTenureDays | 18.1% |
| RegAnciennete | 16.8% |
| TotalTransactions | 8.0% |
| AvgDaysBetweenPurchases | 7.3% |
| TotalQuantity | 7.2% |
| Frequency | 6.8% |
| MonetaryTotal | 6.7% |
| UniqueProducts | 4.7% |
| AvgBasketValue | 3.6% |
| AvgQuantityPerTransaction | 3.0% |

---

## Choix techniques — Anti data leakage

- **28 colonnes supprimées** : ChurnRiskCategory, AccountStatus, 
  LoyaltyLevel, RFMSegment, SatisfactionScore et autres features 
  calculées à partir de Churn
- **Target Encoding Country** : fit sur train uniquement, 
  appliqué sur test
- **StandardScaler** : fit sur X_train uniquement, 
  transform sur X_test
- **Imputation post-split** : médiane calculée sur train, 
  appliquée sur test
- **Pas de re-split** dans train_model.py — on charge 
  directement train.csv et test.csv

---

## Dépendances principales
pandas
numpy
scikit-learn
matplotlib
seaborn
flask
joblib
optuna

---

## Auteur

Projet réalisé dans le cadre de l'atelier Machine Learning — GI2  
Année universitaire 2025-2026