# src/cluster_labels.py

CLUSTER_LABELS = {
    0: "Clients à Risque (Dépenses faibles, Inactifs)",
    1: "Clients Ultra VIP (Exceptions très rentables)",
    2: "Clients Premium/Standard (Clients réguliers)"
}

def interpreter_cluster(cluster_id):
    return CLUSTER_LABELS.get(int(cluster_id), f"Segment {cluster_id}")