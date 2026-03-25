# ============================================================
# src/test.py
# Affichage détaillé des clusters + intervalles
# ============================================================
# PRÉREQUIS : avoir exécuté train_model.py avant
# USAGE     : python test.py
# ============================================================

import pandas as pd
import os

# ============================================================
# CHARGER LES DONNÉES AVEC CLUSTERS
# ============================================================

CHEMIN = '../data/processed/customers_segmented.csv'

if not os.path.exists(CHEMIN):
    raise FileNotFoundError(
        "❌ customers_segmented.csv introuvable !\n"
        "   Lance d'abord : python train_model.py"
    )

df = pd.read_csv(CHEMIN)
print("✅ Dataset chargé :", df.shape)
print(f"   Clusters trouvés : {sorted(df['Cluster'].unique())}")

# ============================================================
# FEATURES UTILISÉES POUR LE CLUSTERING
# ============================================================

features = [
    'Recency',
    'Frequency',
    'MonetaryTotal',
    'CustomerTenureDays',
    'AvgDaysBetweenPurchases',
    'TotalTransactions'
]

# Garder seulement les features présentes
features = [f for f in features if f in df.columns]

# ============================================================
# CALCUL DES INTERVALLES PAR CLUSTER
# ============================================================

cluster_summary = df.groupby('Cluster')[features].agg(['mean', 'min', 'max'])

# ============================================================
# AFFICHAGE PROPRE PAR CLUSTER
# ============================================================

print("\n" + "=" * 55)
print("  ANALYSE DÉTAILLÉE DES CLUSTERS")
print("=" * 55)

def interpreter_cluster(cluster_id, means):
    """Interprétation automatique basée sur les moyennes."""
    recency   = means.get('Recency', 0)
    frequency = means.get('Frequency', 0)
    monetary  = means.get('MonetaryTotal', 0)

    if frequency > 20 and monetary > 3000:
        return "Clients VIP — gros acheteurs fidèles"
    elif recency > 150 and frequency < 5:
        return "Clients à risque — inactifs depuis longtemps"
    elif frequency < 10:
        return "Clients occasionnels — faible engagement"
    else:
        return "Clients fidèles — réguliers et stables"

for cluster_id in cluster_summary.index:

    # Récupérer les valeurs moyennes pour l'interprétation
    moyennes = {
        f: cluster_summary.loc[cluster_id, (f, 'mean')]
        for f in features
    }

    label = interpreter_cluster(cluster_id, moyennes)
    nb_clients = (df['Cluster'] == cluster_id).sum()

    print(f"\n{'=' * 50}")
    print(f"  CLUSTER {cluster_id} — {label}")
    print(f"  Nombre de clients : {nb_clients}")
    print(f"{'=' * 50}")

    for feature in features:
        mean_val = cluster_summary.loc[cluster_id, (feature, 'mean')]
        min_val  = cluster_summary.loc[cluster_id, (feature, 'min')]
        max_val  = cluster_summary.loc[cluster_id, (feature, 'max')]

        print(f"  {feature} :")
        print(f"     Moyenne    : {round(mean_val, 2)}")
        print(f"     Intervalle : [{round(min_val, 2)} → {round(max_val, 2)}]")

# ============================================================
# TAUX DE CHURN PAR CLUSTER (si colonne Churn présente)
# ============================================================

if 'Churn' in df.columns:
    print("\n" + "=" * 55)
    print("  TAUX DE CHURN PAR CLUSTER")
    print("=" * 55)
    churn_par_cluster = df.groupby('Cluster')['Churn'].mean() * 100
    for cluster_id, taux in churn_par_cluster.items():
        print(f"  Cluster {cluster_id} → {taux:.1f}% de churn")

# ============================================================
# SAUVEGARDE CSV PROPRE
# ============================================================

flat_summary = df.groupby('Cluster')[features].mean()

for col in features:
    flat_summary[col + '_min'] = df.groupby('Cluster')[col].min()
    flat_summary[col + '_max'] = df.groupby('Cluster')[col].max()

chemin_csv = '../reports/cluster_intervals_clean.csv'
flat_summary.to_csv(chemin_csv)

print(f"\n✅ Fichier sauvegardé : {chemin_csv}")
print("✅ Analyse des clusters terminée !")