# debug_clustering.py
import pandas as pd
import joblib
import numpy as np

print("=" * 60)
print("  DIAGNOSTIC DU CLUSTERING")
print("=" * 60)

# 1. Charger les modèles
kmeans = joblib.load('../models/kmeans.pkl')
pca = joblib.load('../models/pca.pkl')
scaler_cluster = joblib.load('../models/scaler_cluster.pkl')
cluster_features = joblib.load('../models/cluster_features.pkl')

print(f"\n📊 Features utilisées : {cluster_features}")
print(f"📊 Nombre de clusters : {kmeans.n_clusters}")

# 2. Analyser les centres des clusters (échelle normalisée)
print("\n📊 Centres des clusters (après PCA) :")
centers_pca = kmeans.cluster_centers_
for i, center in enumerate(centers_pca):
    print(f"  Cluster {i}: PC1={center[0]:.3f}, PC2={center[1]:.3f}")

# 3. Tester le petit client
client_petit = pd.DataFrame([{
    'Frequency': 1,
    'MonetaryTotal': 50,
    'CustomerTenureDays': 10,
    'AvgDaysBetweenPurchases': 60,
    'TotalTransactions': 2
}])

print("\n" + "=" * 60)
print("  TEST CLIENT FAIBLE ACTIVITÉ")
print("=" * 60)
print(f"Client : Frequency=1, MonetaryTotal=50€, Tenure=10j")

# Normaliser
X_scaled = scaler_cluster.transform(client_petit[cluster_features].values)
print(f"\nValeurs normalisées :")
for i, col in enumerate(cluster_features):
    print(f"  {col}: {X_scaled[0][i]:.3f}")

# PCA
X_pca = pca.transform(X_scaled)
print(f"\nProjection PCA : PC1={X_pca[0][0]:.3f}, PC2={X_pca[0][1]:.3f}")

# Prédiction
cluster = kmeans.predict(X_pca)[0]
print(f"\n🔴 Cluster prédit : {cluster}")

# 4. Comparer avec les centres des clusters
print("\n📊 Distance aux centres des clusters (en espace PCA) :")
distances = []
for i, center in enumerate(centers_pca):
    dist = np.linalg.norm(X_pca[0] - center)
    distances.append(dist)
    print(f"  Distance au Cluster {i}: {dist:.3f}")
    
print(f"\n✅ Cluster le plus proche : {np.argmin(distances)}")

# 5. Tester un GROS client
client_gros = pd.DataFrame([{
    'Frequency': 50,
    'MonetaryTotal': 8000,
    'CustomerTenureDays': 500,
    'AvgDaysBetweenPurchases': 5,
    'TotalTransactions': 200
}])

print("\n" + "=" * 60)
print("  TEST CLIENT TRÈS ACTIF (VIP)")
print("=" * 60)

X_scaled2 = scaler_cluster.transform(client_gros[cluster_features].values)
X_pca2 = pca.transform(X_scaled2)
cluster2 = kmeans.predict(X_pca2)[0]
print(f"Cluster prédit : {cluster2}")

# 6. Afficher les statistiques des clusters sur les données d'entraînement
print("\n" + "=" * 60)
print("  STATISTIQUES DES CLUSTERS (données réelles)")
print("=" * 60)

df_train = pd.read_csv('../data/train_test/train.csv')

# Ajouter les clusters aux données d'entraînement
df_cluster_data = df_train[cluster_features].copy()
X_scaled_train = scaler_cluster.transform(df_cluster_data.values)
X_pca_train = pca.transform(X_scaled_train)
clusters_train = kmeans.predict(X_pca_train)
df_train['Cluster'] = clusters_train

for cluster_id in sorted(df_train['Cluster'].unique()):
    subset = df_train[df_train['Cluster'] == cluster_id]
    print(f"\n📌 CLUSTER {cluster_id} : {len(subset)} clients ({len(subset)/len(df_train)*100:.1f}%)")
    print(f"   Frequency      : min={subset['Frequency'].min():.0f}, max={subset['Frequency'].max():.0f}, mean={subset['Frequency'].mean():.1f}")
    print(f"   MonetaryTotal  : min={subset['MonetaryTotal'].min():.0f}€, max={subset['MonetaryTotal'].max():.0f}€")
    print(f"   Tenure (jours) : min={subset['CustomerTenureDays'].min():.0f}, max={subset['CustomerTenureDays'].max():.0f}")