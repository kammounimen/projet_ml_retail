# ============================================================
# src/cluster_labels.py
# Labels des clusters — fichier UNIQUE partagé par
# test.py ET app.py
# ============================================================
# ⚠️ Si tu réentraînes et que les clusters changent,
#    modifie UNIQUEMENT ce fichier — les deux scripts
#    se mettront à jour automatiquement.
# ============================================================

CLUSTER_LABELS = {
    0: "Clients peu actifs",
    1: "Clients à risque",
    2: "Clients fidèles",
    3: "Clients VIP",
}

def interpreter_cluster(cluster_id):
    return CLUSTER_LABELS.get(int(cluster_id), f"Segment inconnu (Cluster {cluster_id})")