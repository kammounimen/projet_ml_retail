# Projet ML Retail — Analyse Comportementale Clientèle

Projet de Machine Learning complet sur une base de données e-commerce de cadeaux (4372 clients, 52 features).  
Objectif : segmenter les clients, prédire le churn et estimer le revenu futur.

---

## Résultats obtenus

| Tâche | Algorithme | Résultat |
|-------|-----------|---------|
| Segmentation | KMeans + PCA | 7 clusters — Silhouette = 0.4516 |
| Churn | RandomForest + Optuna | ROC-AUC = 0.9922 — Accuracy = 0.96 |
| Revenu | RandomForest + Optuna | MAE = 562.77 £ — R² = 0.78 |

---

## Description

Ce projet couvre la chaîne complète de traitement en data science :

- Exploration et visualisation des données (valeurs manquantes, outliers, corrélations)
- Nettoyage et préparation des données (preprocessing complet)
- Clustering non supervisé (KMeans + PCA, K optimisé par silhouette score)
- Classification supervisée (prédiction du churn — RandomForest + Optuna)
- Régression supervisée (prédiction du revenu — RandomForest + Optuna)
- Déploiement d'une interface web avec Flask

---

## Structure du projet

```
projet_ml_retail/
├── app/
│   ├── templates/
│   │   └── index.html                  # Interface web (formulaire + résultats)
│   └── app.py                          # Application Flask
├── data/
│   ├── processed/
│   │   ├── customers_segmented.csv     # Données avec labels de clusters
│   │   └── data_clean.csv              # Données nettoyées (86 colonnes)
│   ├── raw/
│   │   └── data_original.csv           # Données brutes (4372 lignes, 52 colonnes)
│   └── train_test/
│       ├── test.csv                    # Données test complètes (875 lignes)
│       ├── train.csv                   # Données train complètes (3497 lignes)
│       ├── X_test.csv                  # Features test
│       ├── X_train.csv                 # Features train
│       ├── y_test.csv                  # Target test
│       └── y_train.csv                 # Target train
├── models/
│   ├── churn_columns.pkl               # Colonnes du modèle churn
│   ├── churn_model.pkl                 # Modèle RandomForest churn
│   ├── cluster_features.pkl            # Features utilisées pour le clustering
│   ├── kmeans.pkl                      # Modèle KMeans
│   ├── pca.pkl                         # Modèle PCA
│   ├── reg_columns.pkl                 # Colonnes du modèle régression
│   ├── regression_model.pkl            # Modèle RandomForest régression
│   ├── scaler_clf.pkl                  # Scaler classification
│   ├── scaler_cluster.pkl              # Scaler clustering
│   └── scaler_reg.pkl                  # Scaler régression
├── notebooks/
│   └── 01_exploration.ipynb            # Exploration initiale des données
├── reports/
│   ├── boxplots_outliers.png           # Boxplots des valeurs aberrantes
│   ├── cluster_analysis.csv            # Statistiques moyennes par cluster
│   ├── cluster_intervals_clean.csv     # Intervalles min/max par cluster
│   ├── clustering_metrics.png          # Silhouette scores par K
│   ├── clusters_2d.png                 # Visualisation 2D des clusters
│   ├── confusion_RandomForest_Churn.png# Matrice de confusion
│   ├── distribution_churn.png          # Distribution de la variable cible
│   ├── feature_importance_churn.csv    # Importance des features (tableau)
│   ├── feature_importance_churn.png    # Importance des features (graphique)
│   ├── matrice_correlation.png         # Heatmap de corrélation
│   ├── pca_2d.png                      # Projection PCA 2D
│   ├── pca_variance.png                # Variance expliquée par composante
│   ├── regression_reel_vs_predit.png   # Réel vs Prédit régression
│   ├── roc_RandomForest_Churn.png      # Courbe ROC
│   └── valeurs_manquantes.png          # Visualisation des NaN
├── src/
│   ├── predict.py                      # Prédiction pour un nouveau client
│   ├── preprocessing.py                # Pipeline complet de nettoyage + split
│   ├── test.py                         # Analyse détaillée des clusters
│   ├── train_model.py                  # Clustering + Classification + Régression
│   └── utils.py                        # Fonctions utilitaires réutilisables
├── .gitignore
├── README.md
└── requirements.txt
```

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
- Parse les dates (RegistrationDate) et les IPs (LastLoginIP)
- Supprime les colonnes inutiles et dangereuses (CustomerID, ChurnRiskCategory...)
- Corrige les valeurs aberrantes (SupportTicketsCount, SatisfactionScore)
- Impute les valeurs manquantes (médiane pour numériques, "Inconnu" pour catégorielles)
- Feature engineering : MonetaryPerDay, AvgBasketValue, TenureRatio, CancelRatio
- Encodage : ordinal + one-hot (Country gérée dans train_model.py)
- Split stratifié 80/20

Résultat : `data/processed/data_clean.csv` + 6 fichiers dans `data/train_test/`

### Étape 2 — Entraînement des modèles

```bash
python train_model.py
```

Ce que ça fait :
- Charge train.csv et test.csv séparément (pas de re-split)
- Target Encoding de Country sur train uniquement (anti data leakage)
- Clustering KMeans + PCA (K=7 optimisé automatiquement par silhouette score)
- Classification churn : RandomForest + Optuna (20 trials)
- Régression revenu : RandomForest + Optuna (20 trials)
- Sauvegarde les 9 modèles dans `models/`
- Génère les graphiques dans `reports/`

Durée : environ 3 à 5 minutes

### Étape 3 — Analyse des clusters

```bash
python test.py
```

Affiche les statistiques détaillées (moyenne, min, max) et le taux de churn par cluster.  
Génère `reports/cluster_intervals_clean.csv`.

### Étape 4 — Test de prédiction

```bash
python predict.py
```

Prédit le cluster, le risque de churn et le revenu estimé pour un client test.

### Étape 5 — Interface web Flask

```bash
cd ../app
python app.py
```

Ouvrir dans le navigateur : [http://127.0.0.1:5000](http://127.0.0.1:5000)

Renseigner les 6 champs et cliquer sur Prédire pour obtenir :
- Le segment client (cluster)
- Le risque de churn (stable ou à risque)
- La probabilité de churn en %
- Le revenu estimé en £

---

## Choix techniques — Anti data leakage

- **Target Encoding Country** : fit sur train uniquement, appliqué sur test
- **StandardScaler** : fit sur X_train uniquement, transform sur X_test
- **Pas de re-split** dans train_model.py — on charge directement train.csv et test.csv
- **Features retirées** de la classification et régression : CancelRatio, TenureRatio, MonetaryPerDay, AvgBasketValue (corrélées au target)

---

## Résultats des clusters

| Cluster | Profil | Nb clients | Taux churn |
|---------|--------|-----------|------------|
| 0 | Clients occasionnels | 1277 | 10.0% |
| 1 | Clients à risque — inactifs | 793 | 100.0% |
| 2 | Clients VIP | 7 | 0.0% |
| 3 | Clients occasionnels | 1230 | 19.5% |
| 4 | Gros acheteurs | 4 | 0.0% |
| 5 | Clients peu actifs | 16 | 12.5% |
| 6 | Clients fidèles | 170 | 0.0% |

---

## Dépendances principales

- pandas, numpy
- scikit-learn
- matplotlib, seaborn
- flask
- joblib
- optuna

---

## Auteur

Projet réalisé dans le cadre de l'atelier Machine Learning — GI2  
Année universitaire 2025-2026