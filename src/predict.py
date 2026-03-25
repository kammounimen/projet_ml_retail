# ============================================================
# src/predict.py
# Prédiction pour un nouveau client
# ============================================================
# PRÉREQUIS : avoir exécuté train_model.py avant
# USAGE     : python predict.py
# ============================================================

import pandas as pd
import joblib
import os

# ============================================================
# CHARGER LES MODÈLES
# ============================================================

CHEMIN_MODELS = '../models'

print("\n" + "=" * 50)
print("  CHARGEMENT DES MODÈLES")
print("=" * 50)

# Clustering
kmeans          = joblib.load(os.path.join(CHEMIN_MODELS, 'kmeans.pkl'))
pca             = joblib.load(os.path.join(CHEMIN_MODELS, 'pca.pkl'))
scaler_cluster  = joblib.load(os.path.join(CHEMIN_MODELS, 'scaler_cluster.pkl'))
cluster_features= joblib.load(os.path.join(CHEMIN_MODELS, 'cluster_features.pkl'))

# Classification
clf             = joblib.load(os.path.join(CHEMIN_MODELS, 'churn_model.pkl'))
scaler_clf      = joblib.load(os.path.join(CHEMIN_MODELS, 'scaler_clf.pkl'))
clf_columns     = joblib.load(os.path.join(CHEMIN_MODELS, 'churn_columns.pkl'))

# Régression
reg             = joblib.load(os.path.join(CHEMIN_MODELS, 'regression_model.pkl'))
scaler_reg      = joblib.load(os.path.join(CHEMIN_MODELS, 'scaler_reg.pkl'))
reg_columns     = joblib.load(os.path.join(CHEMIN_MODELS, 'reg_columns.pkl'))

print("✅ Tous les modèles chargés")


# ============================================================
# INTERPRÉTATION DES CLUSTERS
# ============================================================

def interpreter_cluster(cluster_id):
    interpretations = {
        0: "Clients occasionnels — faible engagement",
        1: "Clients à risque — inactifs depuis longtemps",
        2: "Clients VIP — gros acheteurs fidèles",
        3: "Clients occasionnels — panier moyen",
        4: "Clients gros acheteurs — fréquence élevée",
        5: "Clients peu actifs — à surveiller",
        6: "Clients fidèles — réguliers et stables",
    }
    return interpretations.get(cluster_id, f"Cluster {cluster_id} — non défini")


# ============================================================
# DÉFINIR UN NOUVEAU CLIENT À TESTER
# ============================================================

print("\n" + "=" * 50)
print("  NOUVEAU CLIENT À PRÉDIRE")
print("=" * 50)

# Modifie ces valeurs pour tester différents profils
nouveau_client = pd.DataFrame([{
    'Recency'                 : 15,    # jours depuis dernier achat
    'Frequency'               : 12,   # nombre de commandes
    'MonetaryTotal'           : 850,  # total dépensé en £
    'CustomerTenureDays'      : 400,  # ancienneté en jours
    'AvgDaysBetweenPurchases' : 25,   # délai moyen entre achats
    'TotalTransactions'       : 20,   # nombre total de transactions
}])

print(nouveau_client.to_string(index=False))


# ============================================================
# ÉTAPE 1 : CLUSTERING
# ============================================================

print("\n--- Prédiction Cluster ---")

df_c = nouveau_client.reindex(columns=cluster_features, fill_value=0).astype(float)
X_scaled_c = scaler_cluster.transform(df_c)
X_pca_c    = pca.transform(X_scaled_c)
cluster    = kmeans.predict(X_pca_c)[0]

print(f"  Cluster prédit   : {cluster}")
print(f"  Interprétation   : {interpreter_cluster(cluster)}")


# ============================================================
# ÉTAPE 2 : CLASSIFICATION (Churn)
# ============================================================

print("\n--- Prédiction Churn ---")

df_clf_new = nouveau_client.copy()
df_clf_new = pd.get_dummies(df_clf_new)
df_clf_new = df_clf_new.reindex(columns=clf_columns, fill_value=0)

X_scaled_clf  = scaler_clf.transform(df_clf_new)
churn_pred    = clf.predict(X_scaled_clf)[0]
churn_proba   = clf.predict_proba(X_scaled_clf)[0][1]

if churn_pred == 1:
    statut = "RISQUE D'ATTRITION"
else:
    statut = "CLIENT STABLE"

print(f"  Churn prédit     : {churn_pred} → {statut}")
print(f"  Probabilité churn: {churn_proba:.2%}")


# ============================================================
# ÉTAPE 3 : RÉGRESSION (Revenu estimé)
# ============================================================

print("\n--- Prédiction Revenu ---")

df_reg_new = nouveau_client.copy()
df_reg_new = pd.get_dummies(df_reg_new)
df_reg_new = df_reg_new.reindex(columns=reg_columns, fill_value=0)

X_scaled_reg = scaler_reg.transform(df_reg_new)
revenu_predit = reg.predict(X_scaled_reg)[0]

print(f"  Revenu estimé    : {revenu_predit:.2f} £")


# ============================================================
# RÉSUMÉ FINAL
# ============================================================

print("\n" + "=" * 50)
print("  RÉSUMÉ DE LA PRÉDICTION")
print("=" * 50)
print(f"  Segment client   : Cluster {cluster} — {interpreter_cluster(cluster)}")
print(f"  Risque de churn  : {'OUI' if churn_pred == 1 else 'NON'} ({churn_proba:.2%})")
print(f"  Revenu estimé    : {revenu_predit:.2f} £")
print("=" * 50)
print("\n✅ Prédiction terminée avec succès !")