# ============================================================
# src/train_model.py
# Modélisation complète : Clustering + Classification + Régression
# VERSION FINALE CORRIGÉE
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

# Target Encoding Country
if 'Country' in df_train.columns:
    taux_churn = df_train.groupby('Country')['Churn'].mean()
    df_train['Country_encoded'] = df_train['Country'].map(taux_churn)
    global_mean = df_train['Churn'].mean()
    df_test['Country_encoded'] = df_test['Country'].map(taux_churn).fillna(global_mean)
    df_train = df_train.drop(columns=['Country'])
    df_test  = df_test.drop(columns=['Country'])
    print("✅ Country → Target Encoding (fit train uniquement)")

# ============================================================
# ÉTAPE 2 : CLUSTERING
# ============================================================

print("\n" + "=" * 55)
print("  ÉTAPE 2 : CLUSTERING + PCA")
print("=" * 55)

FEATURES_CLUSTER = [
    'Recency', 'Frequency', 'MonetaryTotal',
    'CustomerTenureDays', 'AvgDaysBetweenPurchases', 'TotalTransactions'
]
FEATURES_CLUSTER = [f for f in FEATURES_CLUSTER if f in df_train.columns]
print(f"Features clustering : {FEATURES_CLUSTER}")

df_cluster = df_train[FEATURES_CLUSTER].copy().fillna(df_train[FEATURES_CLUSTER].mean())

scaler_cluster = StandardScaler()
X_scaled = scaler_cluster.fit_transform(df_cluster)

pca = PCA(n_components=0.95, random_state=42)
X_pca = pca.fit_transform(X_scaled)
print(f"✅ PCA : {pca.n_components_} composantes (95% variance)")

# ── Graphique variance PCA ────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
variance_cumul = np.cumsum(pca.explained_variance_ratio_) * 100
ax.bar(range(1, len(pca.explained_variance_ratio_) + 1),
       pca.explained_variance_ratio_ * 100, color='steelblue')
ax.plot(range(1, len(variance_cumul) + 1), variance_cumul,
        'o-', color='tomato', label='Variance cumulée')
ax.axhline(95, linestyle='--', color='gray', linewidth=0.8)
ax.set_xlabel('Composante')
ax.set_ylabel('Variance (%)')
ax.set_title('Variance expliquée — PCA')
ax.legend()
plt.tight_layout()
sauvegarder('pca_variance.png')

# ── Projection PCA 2D ────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.3, s=10, color='steelblue')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_title('Projection PCA 2D')
plt.tight_layout()
sauvegarder('pca_2d.png')

# ── Recherche du meilleur K ───────────────────────────
print("\nRecherche du meilleur K...")
scores_silhouette = {}
for k in range(2, 10):
    km_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
    score   = silhouette_score(X_pca, km_temp.fit_predict(X_pca))
    scores_silhouette[k] = score
    print(f"  k={k} → silhouette={score:.4f}")

# K forcé à 4 — choix métier (4 segments bien définis)
BEST_K = 4
print(f"\n✅ K forcé à {BEST_K} (choix métier)")

# ── Graphique silhouette avec K=4 ─────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(list(scores_silhouette.keys()),
        list(scores_silhouette.values()), 'o-', color='steelblue')
ax.axvline(BEST_K, linestyle='--', color='tomato', label=f'K={BEST_K}')
ax.set_xlabel('K')
ax.set_ylabel('Silhouette Score')
ax.set_title('Choix du meilleur K')
ax.legend()
plt.tight_layout()
sauvegarder('clustering_metrics.png')

# ── KMeans avec K=4 ──────────────────────────────────
kmeans = KMeans(n_clusters=BEST_K, random_state=42, n_init=10)
df_train['Cluster'] = kmeans.fit_predict(X_pca)

# ── Graphique clusters 2D ─────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1],
                     c=df_train['Cluster'], cmap='tab10', alpha=0.5, s=15)
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_title(f'Clusters KMeans K={BEST_K}')
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

COLS_LEAKAGE = [
    'TenureRatio',
    'AccountStatus_Closed', 'AccountStatus_Suspended',
    'AccountStatus_Pending', 'AccountStatus_Active',
    'CustomerType_Perdu', 'CustomerType_Hyperactif',
    'CustomerType_Regulier', 'CustomerType_Occasionnel', 'CustomerType_Nouveau',
    'LoyaltyLevel',
    'RFMSegment_Dormants', 'RFMSegment_Champions',
    'RFMSegment_Fideles', 'RFMSegment_Potentiels'
]

df_clf_train = df_train.drop(columns=['Cluster'], errors='ignore').copy()
df_clf_train = pd.get_dummies(df_clf_train, drop_first=True)
df_clf_train = df_clf_train.drop(columns=[c for c in COLS_LEAKAGE if c in df_clf_train.columns])

df_clf_test = df_test.copy()
df_clf_test = pd.get_dummies(df_clf_test, drop_first=True)
df_clf_test = df_clf_test.drop(columns=[c for c in COLS_LEAKAGE if c in df_clf_test.columns])

X_train_c = df_clf_train.drop('Churn', axis=1)
y_train_c = df_clf_train['Churn']
X_test_c  = df_clf_test.drop('Churn', axis=1) if 'Churn' in df_clf_test.columns else df_clf_test
y_test_c  = df_clf_test['Churn'] if 'Churn' in df_clf_test.columns else None

X_test_c = X_test_c.reindex(columns=X_train_c.columns, fill_value=0)

print(f"Train : {X_train_c.shape[0]} lignes | Features : {X_train_c.shape[1]}")

scaler_clf   = StandardScaler()
X_train_c_sc = scaler_clf.fit_transform(X_train_c)
X_test_c_sc  = scaler_clf.transform(X_test_c)

print("\nOptimisation Optuna (5 trials)...")
def objective_clf(trial):
    model = RandomForestClassifier(
        n_estimators     = trial.suggest_int('n_estimators', 50, 100),
        max_depth        = trial.suggest_int('max_depth', 3, 10),
        min_samples_split= trial.suggest_int('min_samples_split', 2, 10),
        class_weight     = 'balanced',
        random_state     = 42, n_jobs=1
    )
    return cross_val_score(model, X_train_c_sc, y_train_c, cv=3, scoring='roc_auc').mean()

study_clf = optuna.create_study(direction='maximize')
study_clf.optimize(objective_clf, n_trials=5)
print(f"✅ Meilleurs params churn : {study_clf.best_params}")

clf = RandomForestClassifier(**study_clf.best_params, class_weight='balanced', random_state=42)
clf.fit(X_train_c_sc, y_train_c)

y_pred_c  = clf.predict(X_test_c_sc)
y_proba_c = clf.predict_proba(X_test_c_sc)[:, 1]

if y_test_c is not None:
    print(classification_report(y_test_c, y_pred_c, target_names=['Fidèle (0)', 'Churné (1)']))
    print(f"ROC-AUC : {roc_auc_score(y_test_c, y_proba_c):.4f}")

    # ── Matrice de confusion ──────────────────────────
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

    # ── Courbe ROC ───────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_test_c, y_proba_c, name='RandomForest', ax=ax)
    ax.set_title('Courbe ROC — Churn')
    plt.tight_layout()
    sauvegarder('roc_RandomForest_Churn.png')

# ── Feature importance ────────────────────────────────
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
    'AccountStatus_Closed', 'AccountStatus_Suspended',
    'AccountStatus_Pending', 'AccountStatus_Active',
    'CustomerType_Perdu', 'CustomerType_Hyperactif',
    'CustomerType_Regulier', 'CustomerType_Occasionnel', 'CustomerType_Nouveau',
    'LoyaltyLevel',
    'RFMSegment_Dormants', 'RFMSegment_Champions',
    'RFMSegment_Fideles', 'RFMSegment_Potentiels'
]

df_reg_train = df_train.drop(columns=['Cluster'], errors='ignore').copy()
df_reg_train = pd.get_dummies(df_reg_train, drop_first=True)
df_reg_train = df_reg_train.drop(columns=[c for c in COLS_LEAKAGE_REG if c in df_reg_train.columns])

df_reg_test = df_test.copy()
df_reg_test = pd.get_dummies(df_reg_test, drop_first=True)
df_reg_test = df_reg_test.drop(columns=[c for c in COLS_LEAKAGE_REG if c in df_reg_test.columns])

X_train_r = df_reg_train.drop('AvgBasketValue', axis=1)
y_train_r = df_reg_train['AvgBasketValue']
X_test_r  = df_reg_test.drop('AvgBasketValue', axis=1) if 'AvgBasketValue' in df_reg_test.columns else df_reg_test
y_test_r  = df_reg_test['AvgBasketValue'] if 'AvgBasketValue' in df_reg_test.columns else None

X_test_r = X_test_r.reindex(columns=X_train_r.columns, fill_value=0)

print(f"Train : {X_train_r.shape[0]} lignes | Features : {X_train_r.shape[1]}")
print(f"Cible : AvgBasketValue (panier moyen estimé en £)")

scaler_reg   = StandardScaler()
X_train_r_sc = scaler_reg.fit_transform(X_train_r)
X_test_r_sc  = scaler_reg.transform(X_test_r)

print("\nOptimisation Optuna régression (5 trials)...")
def objective_reg(trial):
    model = RandomForestRegressor(
        n_estimators     = trial.suggest_int('n_estimators', 50, 100),
        max_depth        = trial.suggest_int('max_depth', 3, 10),
        min_samples_split= trial.suggest_int('min_samples_split', 2, 10),
        random_state     = 42, n_jobs=1
    )
    return -cross_val_score(
        model, X_train_r_sc, y_train_r,
        cv=3, scoring='neg_mean_absolute_error'
    ).mean()

study_reg = optuna.create_study(direction='minimize')
study_reg.optimize(objective_reg, n_trials=5)
print(f"✅ Meilleurs params régression : {study_reg.best_params}")

reg = RandomForestRegressor(**study_reg.best_params, random_state=42)
reg.fit(X_train_r_sc, y_train_r)

y_pred_r = reg.predict(X_test_r_sc)
if y_test_r is not None:
    mae = mean_absolute_error(y_test_r, y_pred_r)
    r2  = r2_score(y_test_r, y_pred_r)
    print(f"\nMAE : {mae:.2f} £")
    print(f"R²  : {r2:.4f}")

    # ── Graphique Réel vs Prédit ──────────────────────
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
print("✅ Clustering sauvegardé")

joblib.dump(clf,               os.path.join(CHEMIN_MODELS, 'churn_model.pkl'))
joblib.dump(scaler_clf,        os.path.join(CHEMIN_MODELS, 'scaler_clf.pkl'))
joblib.dump(X_train_c.columns, os.path.join(CHEMIN_MODELS, 'churn_columns.pkl'))
print("✅ Classification sauvegardée")

joblib.dump(reg,               os.path.join(CHEMIN_MODELS, 'regression_model.pkl'))
joblib.dump(scaler_reg,        os.path.join(CHEMIN_MODELS, 'scaler_reg.pkl'))
joblib.dump(X_train_r.columns, os.path.join(CHEMIN_MODELS, 'reg_columns.pkl'))
print("✅ Régression sauvegardée")

# ============================================================
# RÉSUMÉ FINAL
# ============================================================

print("\n" + "=" * 55)
print("  RÉSUMÉ FINAL")
print("=" * 55)
print(f"  Clusters        : {BEST_K}")
if y_test_c is not None:
    print(f"  ROC-AUC churn   : {roc_auc_score(y_test_c, y_proba_c):.4f}")
if y_test_r is not None:
    print(f"  MAE revenu      : {mae:.2f} £")
    print(f"  R² revenu       : {r2:.4f}")
print("\n✅ Lance predict.py ou app.py pour tester !")
print("=" * 55)