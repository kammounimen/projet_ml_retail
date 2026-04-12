# ============================================================
# src/test.py
# Affichage détaillé des clusters + intervalles
# VERSION CORRIGÉE — Labels depuis cluster_labels.py
# ============================================================

import pandas as pd
import os
from cluster_labels import interpreter_cluster, CLUSTER_LABELS

CHEMIN = '../data/processed/customers_segmented.csv'

if not os.path.exists(CHEMIN):
    raise FileNotFoundError(
        "❌ customers_segmented.csv introuvable !\n"
        "   Lance d'abord : python train_model.py"
    )

df = pd.read_csv(CHEMIN)
print("✅ Dataset chargé :", df.shape)
print(f"   Clusters trouvés : {sorted(df['Cluster'].unique())}")

features = [
    'Recency', 'Frequency', 'MonetaryTotal',
    'CustomerTenureDays', 'AvgDaysBetweenPurchases', 'TotalTransactions'
]
features = [f for f in features if f in df.columns]

cluster_summary = df.groupby('Cluster')[features].agg(['mean', 'min', 'max'])

print("\n" + "=" * 55)
print("  ANALYSE DÉTAILLÉE DES CLUSTERS")
print("=" * 55)

for cluster_id in cluster_summary.index:
    label      = interpreter_cluster(cluster_id)
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

if 'Churn' in df.columns:
    print("\n" + "=" * 55)
    print("  TAUX DE CHURN PAR CLUSTER")
    print("=" * 55)
    churn_par_cluster = df.groupby('Cluster')['Churn'].mean() * 100
    for cluster_id, taux in churn_par_cluster.items():
        label = interpreter_cluster(cluster_id)
        print(f"  Cluster {cluster_id} ({label}) → {taux:.1f}% de churn")

flat_summary = df.groupby('Cluster')[features].mean()
for col in features:
    flat_summary[col + '_min'] = df.groupby('Cluster')[col].min()
    flat_summary[col + '_max'] = df.groupby('Cluster')[col].max()

flat_summary.index = [
    f"Cluster {i} — {interpreter_cluster(i)}"
    for i in flat_summary.index
]

chemin_csv = '../reports/cluster_intervals_clean.csv'
flat_summary.to_csv(chemin_csv)
print(f"\n✅ Fichier sauvegardé : {chemin_csv}")
print("✅ Analyse des clusters terminée !")