# ============================================================
# src/predict.py - VERSION CORRIGÉE v3
# ============================================================

import pandas as pd
import joblib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cluster_labels import interpreter_cluster

CHEMIN_MODELS = '../models'

print("\n" + "=" * 50)
print("  CHARGEMENT DES MODÈLES")
print("=" * 50)

# Clustering
kmeans           = joblib.load(os.path.join(CHEMIN_MODELS, 'kmeans.pkl'))
pca              = joblib.load(os.path.join(CHEMIN_MODELS, 'pca.pkl'))
scaler_cluster   = joblib.load(os.path.join(CHEMIN_MODELS, 'scaler_cluster.pkl'))
cluster_features = joblib.load(os.path.join(CHEMIN_MODELS, 'cluster_features.pkl'))

# Classification
clf          = joblib.load(os.path.join(CHEMIN_MODELS, 'churn_model.pkl'))
scaler_clf   = joblib.load(os.path.join(CHEMIN_MODELS, 'scaler_clf.pkl'))
clf_columns  = joblib.load(os.path.join(CHEMIN_MODELS, 'churn_columns.pkl'))

# Régression (chargé mais pas utilisé pour la prédiction finale)
reg          = joblib.load(os.path.join(CHEMIN_MODELS, 'regression_model.pkl'))
scaler_reg   = joblib.load(os.path.join(CHEMIN_MODELS, 'scaler_reg.pkl'))
reg_columns  = joblib.load(os.path.join(CHEMIN_MODELS, 'reg_columns.pkl'))

print("✅ Tous les modèles chargés")


# ============================================================
# NOUVEAU CLIENT À TESTER
# ============================================================

print("\n" + "=" * 50)
print("  NOUVEAU CLIENT À PRÉDIRE")
print("=" * 50)

nouveau_client = pd.DataFrame([{
    'Frequency'               : 12,
    'MonetaryTotal'           : 850,
    'CustomerTenureDays'      : 400,
    'AvgDaysBetweenPurchases' : 25,
    'TotalTransactions'       : 20,
    'TotalQuantity'           : 150,
    'UniqueProducts'          : 30,
    'Age'                     : 45,
    'SupportTicketsCount'     : 1,
}])

print(nouveau_client.to_string(index=False))

# Feature Engineering
nouveau_client['AvgBasketValue'] = (
    nouveau_client['MonetaryTotal'] / (nouveau_client['Frequency'] + 1)
)
nouveau_client['CancelRatio'] = 0.0


# ============================================================
# ÉTAPE 1 : CLUSTERING
# ============================================================

print("\n--- Prédiction Cluster ---")

df_c = nouveau_client[cluster_features].reindex(columns=cluster_features, fill_value=0).astype(float)
X_c_scaled = scaler_cluster.transform(df_c.values)
X_pca_c    = pca.transform(X_c_scaled)
cluster    = kmeans.predict(X_pca_c)[0]

print(f"  Cluster prédit   : {cluster}")
print(f"  Interprétation   : {interpreter_cluster(cluster)}")


# ============================================================
# ÉTAPE 2 : CLASSIFICATION (Churn)
# ============================================================

print("\n--- Prédiction Churn ---")

df_clf_new   = nouveau_client.reindex(columns=clf_columns, fill_value=0)
X_scaled_clf = scaler_clf.transform(df_clf_new)
churn_pred   = clf.predict(X_scaled_clf)[0]
churn_proba  = clf.predict_proba(X_scaled_clf)[0][1]

statut = "RISQUE D'ATTRITION" if churn_pred == 1 else "CLIENT STABLE"
print(f"  Churn prédit     : {churn_pred} → {statut}")
print(f"  Probabilité churn: {churn_proba:.2%}")


# ============================================================
# ÉTAPE 3 : RÉGRESSION (Panier moyen) - CORRIGÉ
# ============================================================

print("\n--- Prédiction Panier Moyen (AvgBasketValue) ---")

# 🔥 CORRECTION : Utiliser le calcul direct (plus fiable)
panier_calcule = nouveau_client['MonetaryTotal'].iloc[0] / (nouveau_client['Frequency'].iloc[0] + 1)
print(f"  Panier moyen estimé : {panier_calcule:.2f} £")


# ============================================================
# RÉSUMÉ FINAL
# ============================================================

print("\n" + "=" * 50)
print("  RÉSUMÉ DE LA PRÉDICTION")
print("=" * 50)
print(f"  Segment client      : Cluster {cluster} — {interpreter_cluster(cluster)}")
print(f"  Risque de churn     : {'OUI' if churn_pred == 1 else 'NON'} ({churn_proba:.2%})")
print(f"  Panier moyen estimé : {panier_calcule:.2f} £")
print("=" * 50)
print("\n✅ Prédiction terminée avec succès !")