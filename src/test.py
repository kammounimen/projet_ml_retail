# ============================================================
# src/test.py - VERSION CORRIGÉE v2
# Analyse les clusters sur les DONNÉES BRUTES (pas normalisées)
# ============================================================

import pandas as pd
import os
import numpy as np
import re
import joblib
from cluster_labels import interpreter_cluster, CLUSTER_LABELS

# ── Chemins ──────────────────────────────────────────────────
CHEMIN_SEGMENTED = '../data/processed/customers_segmented.csv'
CHEMIN_RAW       = '../data/processed/data_clean.csv'

if not os.path.exists(CHEMIN_SEGMENTED):
    raise FileNotFoundError(
        "❌ customers_segmented.csv introuvable !\n"
        "   Lance d'abord : python train_model.py"
    )

# ── Charger les données segmentées (pour récupérer les labels de cluster) ──
df_seg = pd.read_csv(CHEMIN_SEGMENTED)
print("✅ Dataset segmenté chargé :", df_seg.shape)
print(f"   Clusters trouvés : {sorted(df_seg['Cluster'].unique())}")

# ── Charger les données BRUTES pour avoir les vraies valeurs ──
if os.path.exists(CHEMIN_RAW):
    df_raw = pd.read_csv(CHEMIN_RAW)
    print("✅ Données brutes chargées :", df_raw.shape)
    # On recharge les modèles pour re-prédire les clusters sur les données brutes
    kmeans           = joblib.load('../models/kmeans.pkl')
    pca              = joblib.load('../models/pca.pkl')
    scaler_cluster   = joblib.load('../models/scaler_cluster.pkl')
    cluster_features = joblib.load('../models/cluster_features.pkl')

    df_raw_features = df_raw[cluster_features].dropna().copy()
    X_scaled = scaler_cluster.transform(df_raw_features.values)
    X_pca    = pca.transform(X_scaled)
    df_raw_features['Cluster'] = kmeans.predict(X_pca)

    # Joindre avec Churn si disponible
    if 'Churn' in df_raw.columns:
        df_raw_features['Churn'] = df_raw['Churn'].values[:len(df_raw_features)]

    df = df_raw_features
    print("✅ Clusters re-prédits sur données brutes")
else:
    # Fallback : utiliser les données segmentées (normalisées)
    print("⚠️  data_clean.csv introuvable → utilisation de customers_segmented.csv (données normalisées)")
    df = df_seg

features = [f for f in [
    'Frequency', 'MonetaryTotal',
    'CustomerTenureDays', 'AvgDaysBetweenPurchases', 'TotalTransactions'
] if f in df.columns]

print("\n" + "=" * 55)
print("  ANALYSE DÉTAILLÉE DES CLUSTERS (valeurs réelles)")
print("=" * 55)

label_recommendations = {}

for cluster_id in sorted(df['Cluster'].unique()):
    label_actuel = interpreter_cluster(cluster_id)
    nb_clients   = (df['Cluster'] == cluster_id).sum()
    pct_clients  = nb_clients / len(df) * 100

    print(f"\n{'=' * 50}")
    print(f"  CLUSTER {cluster_id} — Label actuel : '{label_actuel}'")
    print(f"  Nombre de clients : {nb_clients} ({pct_clients:.1f}%)")
    print(f"{'=' * 50}")

    stats = {}
    for feature in features:
        mean_val   = df[df['Cluster'] == cluster_id][feature].mean()
        min_val    = df[df['Cluster'] == cluster_id][feature].min()
        max_val    = df[df['Cluster'] == cluster_id][feature].max()
        stats[feature] = {'mean': mean_val, 'min': min_val, 'max': max_val}

        print(f"  {feature} :")
        print(f"     Moyenne    : {round(mean_val, 2)}")
        print(f"     Intervalle : [{round(min_val, 2)} → {round(max_val, 2)}]")

    # ── Analyse automatique (sur vraies valeurs) ──────────────
    print(f"\n  📊 ANALYSE AUTOMATIQUE (valeurs réelles) :")

    freq_mean     = stats['Frequency']['mean']
    monetary_mean = stats['MonetaryTotal']['mean']
    tenure_mean   = stats['CustomerTenureDays']['mean']
    days_between  = stats['AvgDaysBetweenPurchases']['mean']

    # Seuils calculés sur les données BRUTES (pas normalisées)
    freq_q75     = df['Frequency'].quantile(0.75)
    freq_med     = df['Frequency'].median()
    freq_q25     = df['Frequency'].quantile(0.25)
    money_q80    = df['MonetaryTotal'].quantile(0.80)
    money_q75    = df['MonetaryTotal'].quantile(0.75)
    money_q60    = df['MonetaryTotal'].quantile(0.60)
    money_med    = df['MonetaryTotal'].median()
    tenure_q75   = df['CustomerTenureDays'].quantile(0.75)
    tenure_q25   = df['CustomerTenureDays'].quantile(0.25)

    profil = []

    if monetary_mean > money_q75:
        profil.append(f"💰 Dépenses très élevées (>{money_q75:.0f}£, >75e percentile)")
    elif monetary_mean > money_med:
        profil.append(f"💵 Dépenses au-dessus de la médiane ({money_med:.0f}£)")
    else:
        profil.append(f"📉 Dépenses faibles (<{money_med:.0f}£)")

    if freq_mean > freq_q75:
        profil.append(f"🛍️ Fréquence très élevée (>{freq_q75:.1f} cmd, >75e percentile)")
    elif freq_mean > freq_med:
        profil.append(f"🛒 Fréquence au-dessus de la médiane ({freq_med:.1f} cmd)")
    else:
        profil.append(f"📆 Fréquence faible (<{freq_med:.1f} cmd)")

    if tenure_mean > tenure_q75:
        profil.append(f"⭐ Clients très anciens (>{tenure_q75:.0f} jours)")
    elif tenure_mean < tenure_q25:
        profil.append(f"🆕 Nouveaux clients (<{tenure_q25:.0f} jours)")
    else:
        profil.append(f"📅 Ancienneté moyenne ({tenure_mean:.0f} jours)")

    if 'Churn' in df.columns:
        churn_rate = df[df['Cluster'] == cluster_id]['Churn'].mean() * 100
        stats['churn_rate'] = churn_rate
        if churn_rate > 50:
            profil.append(f"⚠️ TRÈS HAUT RISQUE churn ({churn_rate:.1f}%)")
        elif churn_rate > 30:
            profil.append(f"⚠️ Risque churn modéré ({churn_rate:.1f}%)")
        elif churn_rate > 10:
            profil.append(f"✅ Risque churn faible ({churn_rate:.1f}%)")
        else:
            profil.append(f"🟢 Risque churn très faible ({churn_rate:.1f}%)")

    for p in profil:
        print(f"     • {p}")

    # ── Recommandation de label ───────────────────────────────
    print(f"\n  💡 RECOMMANDATION DE LABEL :")

    # On isole d'abord les valeurs extrêmes (Cluster 1)
    if monetary_mean > money_q80 * 10: 
        recommended_label = "👑 Ultra-VIP (Exceptions)"
        if 'Churn' in df.columns and stats.get('churn_rate', 0) > 40:
            recommended_label += " ⚠️ HAUT RISQUE CHURN"
            
    # Ensuite les très bons clients (Cluster 2)
    elif monetary_mean > money_q75 and freq_mean > freq_med:
        recommended_label = "⭐ Clients Premium (Réguliers)"
        
    # Et enfin les autres (Cluster 0)
    elif freq_mean < freq_q75 and monetary_mean < money_med * 2:
        recommended_label = "📉 Clients Standard / Faible activité"
        
    elif 'Churn' in df.columns:
        churn_rate = df[df['Cluster'] == cluster_id]['Churn'].mean() * 100
        if churn_rate > 40:
            recommended_label = "⚠️ Clients à Risque Élevé (churn >40%)"
        else:
            recommended_label = "📊 Clients à Surveiller"
    else:
        recommended_label = f"Segment {cluster_id} (analyser les stats)"

    print(f"     → {recommended_label}")

    label_recommendations[cluster_id] = {
        'current_label'   : label_actuel,
        'recommended_label': recommended_label,
        'stats'           : stats
    }

# ── Taux de churn par cluster ─────────────────────────────────
if 'Churn' in df.columns:
    print("\n" + "=" * 55)
    print("  TAUX DE CHURN PAR CLUSTER")
    print("=" * 55)
    churn_par_cluster = df.groupby('Cluster')['Churn'].mean() * 100
    for cluster_id, taux in churn_par_cluster.items():
        label = interpreter_cluster(cluster_id)
        print(f"  Cluster {cluster_id} ({label}) → {taux:.1f}% de churn")

# ── Sauvegarder les statistiques ─────────────────────────────
flat_summary = df.groupby('Cluster')[features].mean()
for col in features:
    flat_summary[col + '_min'] = df.groupby('Cluster')[col].min()
    flat_summary[col + '_max'] = df.groupby('Cluster')[col].max()

if 'Churn' in df.columns:
    flat_summary['Churn_Rate_%'] = df.groupby('Cluster')['Churn'].mean() * 100

flat_summary.index = [
    f"Cluster {i} — {interpreter_cluster(i)}"
    for i in flat_summary.index
]

chemin_csv = '../reports/cluster_intervals_clean.csv'
flat_summary.to_csv(chemin_csv)
print(f"\n✅ Fichier sauvegardé : {chemin_csv}")

# ── Génération des labels corrigés ───────────────────────────
print("\n" + "=" * 55)
print("  GÉNÉRATION DES LABELS CORRIGÉS")
print("=" * 55)

new_labels = {}
for cluster_id, rec in label_recommendations.items():
    recommended = rec['recommended_label']
    clean_label = re.sub(r'[^\w\s\(\)\-,/]', '', recommended).strip()
    new_labels[cluster_id] = clean_label

print("\n📝 Labels recommandés pour cluster_labels.py :")
print("\nCLUSTER_LABELS = {")
for cluster_id, label in sorted(new_labels.items()):
    print(f'    {cluster_id}: "{label}",')
print("}")

reco_path = '../reports/recommended_labels.txt'
os.makedirs('../reports', exist_ok=True)
with open(reco_path, 'w', encoding='utf-8') as f:
    f.write("# Labels recommandés pour cluster_labels.py\n")
    f.write("# Basés sur les VALEURS RÉELLES (données brutes non normalisées)\n\n")
    f.write("CLUSTER_LABELS = {\n")
    for cluster_id, label in sorted(new_labels.items()):
        f.write(f'    {cluster_id}: "{label}",\n')
    f.write("}\n\n")
    f.write("# Statistiques par cluster (valeurs réelles) :\n")
    for cluster_id, rec in label_recommendations.items():
        stats = rec['stats']
        f.write(f"\n# Cluster {cluster_id} :\n")
        f.write(f"#   Frequency moyenne      : {stats['Frequency']['mean']:.1f} commandes\n")
        f.write(f"#   MonetaryTotal moyenne  : {stats['MonetaryTotal']['mean']:.0f} £\n")
        f.write(f"#   CustomerTenureDays moy : {stats['CustomerTenureDays']['mean']:.0f} jours\n")
        if 'churn_rate' in stats:
            f.write(f"#   Taux de churn          : {stats['churn_rate']:.1f}%\n")

print(f"\n✅ Recommandations sauvegardées dans : {reco_path}")
print("=" * 55)
print("✅ Analyse des clusters terminée ! (basée sur données réelles)")