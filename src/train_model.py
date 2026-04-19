# ============================================================
# src/train_model.py
# Modélisation complète : Clustering + Classification + Régression
# VERSION CORRIGÉE v4
# ============================================================

import pandas as pd
import numpy as np
import os
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
import json

optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    RocCurveDisplay,
    mean_absolute_error,
    r2_score
)

# ============================================================
# CHEMINS
# ============================================================

CHEMIN_TRAIN   = '../data/train_test/train.csv'
CHEMIN_TEST    = '../data/train_test/test.csv'
CHEMIN_REPORTS = '../reports'
CHEMIN_MODELS  = '../models'
CHEMIN_DATA    = '../data/processed'

os.makedirs(CHEMIN_REPORTS, exist_ok=True)
os.makedirs(CHEMIN_MODELS,  exist_ok=True)
os.makedirs(CHEMIN_DATA,    exist_ok=True)


def sauvegarder(nom):
    chemin = os.path.join(CHEMIN_REPORTS, nom)
    plt.savefig(chemin, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  ✅ Graphique sauvegardé → {chemin}")


# ============================================================
# ÉTAPE 1 : Charger train ET test
# ============================================================

print("\n" + "=" * 55)
print("  CHARGEMENT DES DONNÉES")
print("=" * 55)

df_train = pd.read_csv(CHEMIN_TRAIN)
df_test  = pd.read_csv(CHEMIN_TEST)

print(f"✅ Train : {df_train.shape[0]} lignes × {df_train.shape[1]} colonnes")
print(f"✅ Test  : {df_test.shape[0]}  lignes × {df_test.shape[1]}  colonnes")

if 'Churn' not in df_train.columns:
    raise ValueError("❌ Colonne 'Churn' introuvable dans train.csv !")

if 'Country' in df_train.columns:
    df_train = df_train.drop(columns=['Country'])
    df_test  = df_test.drop(columns=['Country'])
    print("Country supprimee (deja traitee dans preprocessing.py)")


# ============================================================
# ÉTAPE 2 : CLUSTERING - VERSION CORRIGÉE
# ============================================================

print("\n" + "=" * 55)
print("  ÉTAPE 2 : CLUSTERING + PCA (VERSION CORRIGÉE)")
print("=" * 55)

FEATURES_CLUSTER = [
    'Frequency', 'MonetaryTotal',
    'CustomerTenureDays', 'AvgDaysBetweenPurchases', 'TotalTransactions'
]
FEATURES_CLUSTER = [f for f in FEATURES_CLUSTER if f in df_train.columns]
print(f"Features clustering : {FEATURES_CLUSTER}")

df_cluster = df_train[FEATURES_CLUSTER].copy()

# S'assurer que les données sont en float
df_cluster = df_cluster.astype(float)

# Combler les NaN avec la médiane
for col in FEATURES_CLUSTER:
    if df_cluster[col].isnull().any():
        mediane = df_cluster[col].median()
        df_cluster[col] = df_cluster[col].fillna(mediane)
        print(f"  NaN residuels dans {col} -> mediane ({mediane:.2f})")

# StandardScaler avec vérification
scaler_cluster = StandardScaler()
X_cluster_scaled = scaler_cluster.fit_transform(df_cluster)

# Vérifier que la normalisation a fonctionné
print(f"✅ Scaler cluster entraîné")
print(f"   Moyennes après normalisation : {X_cluster_scaled.mean(axis=0).round(6)}")
print(f"   Écarts-types après normalisation : {X_cluster_scaled.std(axis=0).round(6)}")

# PCA avec conservation de 95% de variance
pca = PCA(n_components=0.95, random_state=42)
X_pca = pca.fit_transform(X_cluster_scaled)
print(f"✅ PCA : {pca.n_components_} composantes conservées")
print(f"   Variance expliquée : {pca.explained_variance_ratio_.sum()*100:.1f}%")

# Graphique variance PCA
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(range(1, len(pca.explained_variance_ratio_)+1),
       pca.explained_variance_ratio_ * 100,
       color='steelblue')
ax.set_xlabel('Composante principale')
ax.set_ylabel('Variance expliquée (%)')
ax.set_title(f'Variance PCA - Total : {pca.explained_variance_ratio_.sum()*100:.1f}%')
plt.tight_layout()
sauvegarder('pca_variance.png')

# Projection PCA 2D (pour visualisation)
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.3, s=10, color='steelblue')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_title('Projection PCA 2D (données clustering)')
plt.tight_layout()
sauvegarder('pca_2d.png')

# Recherche du K optimal (contrainte assouplie)
print("\nRecherche du K optimal (silhouette score)...")

MIN_CLUSTER_RATIO = 0.01
scores_silhouette = {}

for k in range(2, 7):
    km_test = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels_test = km_test.fit_predict(X_pca)
    
    counts = pd.Series(labels_test).value_counts(normalize=True)
    min_ratio = counts.min()
    
    if min_ratio < MIN_CLUSTER_RATIO:
        print(f"  K={k} : rejeté (cluster trop petit : {min_ratio:.1%})")
        continue
    
    score = silhouette_score(X_pca, labels_test)
    scores_silhouette[k] = score
    print(f"  K={k} : silhouette={score:.4f} | plus petit cluster={min_ratio:.1%}")

if not scores_silhouette:
    print("  Aucun K valide trouvé → fallback K=3")
    BEST_K = 3
else:
    BEST_K = max(scores_silhouette, key=scores_silhouette.get)
    print(f"✅ K optimal choisi : {BEST_K} (silhouette={scores_silhouette[BEST_K]:.4f})")

# Graphique silhouette
if scores_silhouette:
    fig, ax = plt.subplots(figsize=(7, 4))
    ks = list(scores_silhouette.keys())
    scores = list(scores_silhouette.values())
    ax.plot(ks, scores, 'o-', color='steelblue', linewidth=2, markersize=8)
    ax.axvline(BEST_K, linestyle='--', color='tomato', label=f'K optimal = {BEST_K}')
    ax.set_xlabel('Nombre de clusters K')
    ax.set_ylabel('Score Silhouette')
    ax.set_title('Choix du K optimal — Score Silhouette')
    ax.legend()
    plt.tight_layout()
    sauvegarder('silhouette_scores.png')

# KMeans final
kmeans = KMeans(n_clusters=BEST_K, random_state=42, n_init=10)
df_train['Cluster'] = kmeans.fit_predict(X_pca)

# Afficher la distribution des clusters
print("\nDistribution des clusters :")
cluster_counts = df_train['Cluster'].value_counts().sort_index()
for c, n in cluster_counts.items():
    print(f"  Cluster {c} : {n} clients ({n/len(df_train):.1%})")

# Graphique clusters 2D
fig, ax = plt.subplots(figsize=(8, 5))
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1],
                     c=df_train['Cluster'], cmap='tab10', alpha=0.5, s=15)
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_title(f'Clusters KMeans K={BEST_K} (Silhouette optimisé)')
plt.colorbar(scatter, ax=ax, label='Cluster')
plt.tight_layout()
sauvegarder('clusters_2d.png')

df_train.to_csv(os.path.join(CHEMIN_DATA, 'customers_segmented.csv'), index=False)
print("✅ customers_segmented.csv sauvegardé")


# ============================================================
# ÉTAPE 3 : CLASSIFICATION CHURN
# ============================================================

print("\n" + "=" * 55)
print("  ÉTAPE 3 : CLASSIFICATION CHURN")
print("=" * 55)

COLS_LEAKAGE_CLF = [
    'TenureRatio',
    'LoyaltyLevel',
    'MonetaryPerDay',
    'MonetaryAvg',
    'MonetaryMin',
    'MonetaryMax',
    'MonetaryStd',
]

df_clf_train = df_train.drop(columns=['Cluster'], errors='ignore').copy()
cols_to_drop_train = [c for c in df_clf_train.columns if c in COLS_LEAKAGE_CLF]
df_clf_train = df_clf_train.drop(columns=cols_to_drop_train, errors='ignore')

df_clf_test = df_test.copy()
cols_to_drop_test = [c for c in df_clf_test.columns if c in COLS_LEAKAGE_CLF]
df_clf_test = df_clf_test.drop(columns=cols_to_drop_test, errors='ignore')

X_train_c = df_clf_train.drop('Churn', axis=1)
y_train_c = df_clf_train['Churn']
X_test_c  = df_clf_test.drop('Churn', axis=1) if 'Churn' in df_clf_test.columns else df_clf_test
y_test_c  = df_clf_test['Churn'] if 'Churn' in df_clf_test.columns else None

X_test_c = X_test_c.reindex(columns=X_train_c.columns, fill_value=0)

print(f"Train : {X_train_c.shape[0]} lignes | Features : {X_train_c.shape[1]}")

scaler_clf   = StandardScaler()
X_train_c_sc = scaler_clf.fit_transform(X_train_c)
X_test_c_sc  = scaler_clf.transform(X_test_c)

print("\nOptimisation Optuna (30 trials)...")
def objective_clf(trial):
    model = RandomForestClassifier(
        n_estimators     = trial.suggest_int('n_estimators', 50, 300),
        max_depth        = trial.suggest_int('max_depth', 3, 15),
        min_samples_split= trial.suggest_int('min_samples_split', 2, 20),
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10),
        class_weight     = 'balanced',
        random_state     = 42, n_jobs=-1
    )
    return cross_val_score(model, X_train_c_sc, y_train_c, cv=5, scoring='roc_auc').mean()

study_clf = optuna.create_study(direction='maximize')
study_clf.optimize(objective_clf, n_trials=30)
print(f"✅ Meilleurs params churn : {study_clf.best_params}")

clf = RandomForestClassifier(**study_clf.best_params, class_weight='balanced', random_state=42)
clf.fit(X_train_c_sc, y_train_c)

y_pred_c  = clf.predict(X_test_c_sc)
y_proba_c = clf.predict_proba(X_test_c_sc)[:, 1]

if y_test_c is not None:
    print(classification_report(y_test_c, y_pred_c, target_names=['Fidèle (0)', 'Churné (1)']))
    print(f"ROC-AUC : {roc_auc_score(y_test_c, y_proba_c):.4f}")

    cm = confusion_matrix(y_test_c, y_pred_c)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Fidèle', 'Churné'],
                yticklabels=['Fidèle', 'Churné'], ax=ax)
    ax.set_title('Matrice de confusion — Churn')
    ax.set_ylabel('Réel')
    ax.set_xlabel('Prédit')
    plt.tight_layout()
    sauvegarder('confusion_RandomForest_Churn.png')

    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_test_c, y_proba_c, name='RandomForest', ax=ax)
    ax.set_title('Courbe ROC — Churn')
    plt.tight_layout()
    sauvegarder('roc_RandomForest_Churn.png')

importances = pd.Series(clf.feature_importances_,
                         index=X_train_c.columns).sort_values(ascending=False)
print("\n🔍 Top 10 features les plus importantes :")
print(importances.head(10))
importances.to_csv(os.path.join(CHEMIN_REPORTS, 'feature_importance_churn.csv'))

fig, ax = plt.subplots(figsize=(10, 6))
importances.head(20).plot(kind='bar', color='steelblue', ax=ax)
ax.set_title('Top 20 features — Churn')
ax.set_ylabel('Importance')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
sauvegarder('feature_importance_churn.png')


# ============================================================
# ÉTAPE 4 : RÉGRESSION — PANIER MOYEN (AvgBasketValue)
# ============================================================

print("\n" + "=" * 55)
print("  ÉTAPE 4 : RÉGRESSION PANIER MOYEN (AvgBasketValue)")
print("=" * 55)

if 'AvgBasketValue' not in df_train.columns:
    df_train['AvgBasketValue'] = df_train['MonetaryTotal'] / (df_train['Frequency'] + 1)
if 'AvgBasketValue' not in df_test.columns:
    df_test['AvgBasketValue']  = df_test['MonetaryTotal']  / (df_test['Frequency']  + 1)

COLS_LEAKAGE_REG = [
    'Churn',
    'TenureRatio',
    'LoyaltyLevel',
    'MonetaryPerDay',
    'MonetaryAvg',
    'MonetaryMin',
    'MonetaryMax',
    'MonetaryStd',
]

df_reg_train = df_train.drop(columns=['Cluster'], errors='ignore').copy()
cols_to_drop_train_reg = [c for c in df_reg_train.columns if c in COLS_LEAKAGE_REG]
df_reg_train = df_reg_train.drop(columns=cols_to_drop_train_reg, errors='ignore')

df_reg_test = df_test.copy()
cols_to_drop_test_reg = [c for c in df_reg_test.columns if c in COLS_LEAKAGE_REG]
df_reg_test = df_reg_test.drop(columns=cols_to_drop_test_reg, errors='ignore')

X_train_r = df_reg_train.drop('AvgBasketValue', axis=1)
y_train_r = df_reg_train['AvgBasketValue']
X_test_r  = df_reg_test.drop('AvgBasketValue', axis=1) if 'AvgBasketValue' in df_reg_test.columns else df_reg_test
y_test_r  = df_reg_test['AvgBasketValue'] if 'AvgBasketValue' in df_reg_test.columns else None

X_test_r = X_test_r.reindex(columns=X_train_r.columns, fill_value=0)

print(f"Train : {X_train_r.shape[0]} lignes | Features : {X_train_r.shape[1]}")
print(f"Cible : AvgBasketValue (panier moyen estimé en £)")
print(f"🔍 Statistiques de la cible : min={y_train_r.min():.2f}, max={y_train_r.max():.2f}, mean={y_train_r.mean():.2f}")

scaler_reg   = StandardScaler()
X_train_r_sc = scaler_reg.fit_transform(X_train_r)
X_test_r_sc  = scaler_reg.transform(X_test_r)

print("\nOptimisation Optuna régression (30 trials)...")
def objective_reg(trial):
    model = RandomForestRegressor(
        n_estimators     = trial.suggest_int('n_estimators', 50, 300),
        max_depth        = trial.suggest_int('max_depth', 3, 15),
        min_samples_split= trial.suggest_int('min_samples_split', 2, 20),
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10),
        random_state     = 42, n_jobs=-1
    )
    return -cross_val_score(
        model, X_train_r_sc, y_train_r,
        cv=5, scoring='neg_mean_absolute_error'
    ).mean()

study_reg = optuna.create_study(direction='minimize')
study_reg.optimize(objective_reg, n_trials=30)
print(f"✅ Meilleurs params régression : {study_reg.best_params}")

reg = RandomForestRegressor(**study_reg.best_params, random_state=42)
reg.fit(X_train_r_sc, y_train_r)

y_pred_r = reg.predict(X_test_r_sc)
if y_test_r is not None:
    mae = mean_absolute_error(y_test_r, y_pred_r)
    r2  = r2_score(y_test_r, y_pred_r)
    print(f"\nMAE : {mae:.2f} £")
    print(f"R²  : {r2:.4f}")

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_test_r, y_pred_r, alpha=0.4, s=15, color='steelblue')
    lim_min = min(y_test_r.min(), y_pred_r.min())
    lim_max = max(y_test_r.max(), y_pred_r.max())
    ax.plot([lim_min, lim_max], [lim_min, lim_max], '--', color='tomato', linewidth=1)
    ax.set_xlabel('Réel (£)')
    ax.set_ylabel('Prédit (£)')
    ax.set_title(f'Réel vs Prédit\nMAE={mae:.1f} £  R²={r2:.3f}')
    plt.tight_layout()
    sauvegarder('regression_reel_vs_predit.png')
    
    # 🔥 Vérification supplémentaire
    print("\n🔍 Vérification des prédictions :")
    print(f"   Moyenne des prédictions : {y_pred_r.mean():.2f} £")
    print(f"   Moyenne des valeurs réelles : {y_test_r.mean():.2f} £")
    if y_pred_r.mean() < y_test_r.mean() * 0.5:
        print("   ⚠️ ATTENTION : Les prédictions sont beaucoup trop petites !")
        print("   → Utilisez le calcul direct dans app.py au lieu du modèle ML")


# ============================================================
# ÉTAPE 5 : SAUVEGARDER LES MODÈLES
# ============================================================

print("\n" + "=" * 55)
print("  ÉTAPE 5 : SAUVEGARDE DES MODÈLES")
print("=" * 55)

joblib.dump(kmeans,            os.path.join(CHEMIN_MODELS, 'kmeans.pkl'))
joblib.dump(pca,               os.path.join(CHEMIN_MODELS, 'pca.pkl'))
joblib.dump(scaler_cluster,    os.path.join(CHEMIN_MODELS, 'scaler_cluster.pkl'))
joblib.dump(FEATURES_CLUSTER,  os.path.join(CHEMIN_MODELS, 'cluster_features.pkl'))
print("✅ Clustering sauvegardé (avec scaler_cluster.pkl)")

joblib.dump(clf,               os.path.join(CHEMIN_MODELS, 'churn_model.pkl'))
joblib.dump(scaler_clf,        os.path.join(CHEMIN_MODELS, 'scaler_clf.pkl'))
joblib.dump(X_train_c.columns, os.path.join(CHEMIN_MODELS, 'churn_columns.pkl'))
print("✅ Classification sauvegardée")

joblib.dump(reg,               os.path.join(CHEMIN_MODELS, 'regression_model.pkl'))
joblib.dump(scaler_reg,        os.path.join(CHEMIN_MODELS, 'scaler_reg.pkl'))
joblib.dump(X_train_r.columns, os.path.join(CHEMIN_MODELS, 'reg_columns.pkl'))
print("✅ Régression sauvegardée")

metriques = {
    "roc_auc"  : round(roc_auc_score(y_test_c, y_proba_c), 4) if y_test_c is not None else None,
    "accuracy" : f"{round(clf.score(X_test_c_sc, y_test_c) * 100, 1)}%" if y_test_c is not None else None,
    "mae"      : round(mae, 2) if y_test_r is not None else None,
    "r2"       : round(r2, 4) if y_test_r is not None else None,
    "best_k"   : BEST_K,
}
 
metrics_path = os.path.join(CHEMIN_REPORTS, 'model_metrics.json')
with open(metrics_path, 'w') as f:
    json.dump(metriques, f, indent=2)
 
print(f"✅ Métriques sauvegardées → {metrics_path}")


# ============================================================
# RÉSUMÉ FINAL
# ============================================================

print("\n" + "=" * 55)
print("  RÉSUMÉ FINAL")
print("=" * 55)
print(f"  Clusters        : {BEST_K} (K optimal silhouette)")
if scores_silhouette:
    print(f"  Score silhouette: {scores_silhouette[BEST_K]:.4f}")
print("\nDistribution clusters (équilibre) :")
for c, n in cluster_counts.items():
    print(f"  Cluster {c} : {n} clients ({n/len(df_train):.1%})")
if y_test_c is not None:
    print(f"\n  ROC-AUC churn   : {roc_auc_score(y_test_c, y_proba_c):.4f}")
if y_test_r is not None:
    print(f"  MAE panier moyen: {mae:.2f} £")
    print(f"  R² panier moyen : {r2:.4f}")
print("\n✅ Lance predict.py ou app.py pour tester !")
print("=" * 55)