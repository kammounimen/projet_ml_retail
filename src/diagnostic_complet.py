# diagnostic_complet.py
import pandas as pd
import joblib
import numpy as np

print("=" * 60)
print("  DIAGNOSTIC COMPLET DES CLUSTERS (Données brutes)")
print("=" * 60)

# 1. Charger les données BRUTES (avant normalisation)
df_raw = pd.read_csv('../data/processed/data_clean.csv')
print(f"\n✅ Données brutes chargées : {df_raw.shape}")

# 2. Charger les clusters depuis le modèle
df_train = pd.read_csv('../data/train_test/train.csv')
print(f"✅ Train data : {df_train.shape}")

# 3. Recharger le modèle KMeans pour prédire les clusters sur les données brutes
kmeans = joblib.load('../models/kmeans.pkl')
pca = joblib.load('../models/pca.pkl')
scaler_cluster = joblib.load('../models/scaler_cluster.pkl')
cluster_features = joblib.load('../models/cluster_features.pkl')

print(f"\n📊 Features utilisées pour le clustering : {cluster_features}")

# 4. Prendre un échantillon des données brutes pour analyse
df_sample = df_raw[cluster_features].dropna().copy()

# Normaliser et prédire
X_scaled = scaler_cluster.transform(df_sample.values)
X_pca = pca.transform(X_scaled)
clusters = kmeans.predict(X_pca)

# Ajouter les clusters aux données brutes
df_sample['Cluster'] = clusters

print("\n" + "=" * 60)
print("  ANALYSE DES CLUSTERS (VALEURS RÉELLES)")
print("=" * 60)

for cluster_id in sorted(df_sample['Cluster'].unique()):
    subset = df_sample[df_sample['Cluster'] == cluster_id]
    nb_clients = len(subset)
    pct = nb_clients / len(df_sample) * 100
    
    print(f"\n{'=' * 50}")
    print(f"  CLUSTER {cluster_id}")
    print(f"  Nombre de clients : {nb_clients} ({pct:.1f}%)")
    print(f"{'=' * 50}")
    
    for feature in cluster_features:
        mean_val = subset[feature].mean()
        min_val = subset[feature].min()
        max_val = subset[feature].max()
        median_val = subset[feature].median()
        
        print(f"\n  {feature} :")
        print(f"     Min     : {min_val:.2f}")
        print(f"     Max     : {max_val:.2f}")
        print(f"     Moyenne : {mean_val:.2f}")
        print(f"     Médiane : {median_val:.2f}")
    
    # Taux de churn si disponible
    if 'Churn' in df_raw.columns and len(subset) <= len(df_raw):
        # Essayer de joindre avec les données originales
        pass

# 5. Recommandations basées sur les VRAIES valeurs
print("\n" + "=" * 60)
print("  RECOMMANDATIONS DE LABELS (Basées sur valeurs réelles)")
print("=" * 60)

for cluster_id in sorted(df_sample['Cluster'].unique()):
    subset = df_sample[df_sample['Cluster'] == cluster_id]
    
    freq_mean = subset['Frequency'].mean()
    monetary_mean = subset['MonetaryTotal'].mean()
    tenure_mean = subset['CustomerTenureDays'].mean()
    days_between_mean = subset['AvgDaysBetweenPurchases'].mean()
    
    print(f"\n📌 Cluster {cluster_id} :")
    print(f"   Frequency moyenne : {freq_mean:.1f} commandes")
    print(f"   MonetaryTotal moyenne : {monetary_mean:.0f} €")
    print(f"   Ancienneté moyenne : {tenure_mean:.0f} jours")
    print(f"   Délai moyen entre achats : {days_between_mean:.1f} jours")
    
    # Déterminer le label
    if monetary_mean > 2000 and freq_mean > 20:
        label = "👑 Clients Élite VIP"
    elif monetary_mean > 1000 and freq_mean > 10:
        label = "⭐ Clients Premium"
    elif freq_mean < 5 and monetary_mean < 500:
        label = "🆕 Clients Occasionnels"
    elif tenure_mean > 500 and freq_mean > 5:
        label = "🏆 Clients Fidèles Historiques"
    else:
        label = "📊 Clients Standards"
    
    print(f"   → Label recommandé : {label}")

# 6. Sauvegarder les résultats
df_sample.to_csv('../reports/clusters_with_real_values.csv', index=False)
print("\n✅ Résultats sauvegardés dans : reports/clusters_with_real_values.csv")