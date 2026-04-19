from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
import os
import sys
import json

# Ajout du chemin vers src pour importer interpreter_cluster
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from cluster_labels import interpreter_cluster

app = Flask(__name__)

CHEMIN_MODELS = '../models'
CHEMIN_REPORTS = '../reports'

# --- CHARGEMENT DES MODÈLES ---
kmeans = joblib.load(os.path.join(CHEMIN_MODELS, 'kmeans.pkl'))
pca = joblib.load(os.path.join(CHEMIN_MODELS, 'pca.pkl'))
scaler_cluster = joblib.load(os.path.join(CHEMIN_MODELS, 'scaler_cluster.pkl'))
cluster_features = joblib.load(os.path.join(CHEMIN_MODELS, 'cluster_features.pkl'))

clf = joblib.load(os.path.join(CHEMIN_MODELS, 'churn_model.pkl'))
scaler_clf = joblib.load(os.path.join(CHEMIN_MODELS, 'scaler_clf.pkl'))
clf_columns = joblib.load(os.path.join(CHEMIN_MODELS, 'churn_columns.pkl'))

# --- CHARGEMENT DES MÉTRIQUES ---
metrics_path = os.path.join(CHEMIN_REPORTS, 'model_metrics.json')
MODEL_METRICS = {}
if os.path.exists(metrics_path):
    with open(metrics_path, 'r') as f:
        MODEL_METRICS = json.load(f)

def get_risk_label(proba: float) -> tuple:
    """Retourne le niveau de risque, sa couleur CSS et son libellé."""
    if proba < 30:
        return "Faible", "green", "Stable"
    elif proba < 50:
        return "Modéré", "orange", "À surveiller"
    else:
        return "Élevé", "red", "À risque"

def get_recommendations(proba: float, segment: str) -> list:
    """Génère des conseils cohérents avec le risque de churn."""
    if proba >= 50:
        return [
            "🚨 Offre de rétention urgente : Envoyer un coupon de réduction immédiatement.",
            "📞 Appel de courtoisie : Identifier la cause de l'insatisfaction (tickets support).",
            "💸 Geste commercial : Proposer un remboursement ou un avantage sur la prochaine commande."
        ]
    elif proba >= 30:
        return [
            "📧 Email de réengagement : Rappeler les nouveautés du catalogue.",
            "🛍️ Recommandations personnalisées : Proposer des produits similaires à ses achats.",
            "👀 Monitoring : Surveiller l'activité sur les 15 prochains jours."
        ]
    else:
        return [
            "⭐ Fidélisation : Inclure le client dans le programme Ambassadeur.",
            "🎁 Récompense : Offrir un accès anticipé aux prochaines ventes privées.",
            "📊 Cross-sell : Suggérer des produits premium complémentaires."
        ]

@app.route("/", methods=["GET", "POST"])
def index():
    template_globals = {
        'form_data': {},
        'best_k': kmeans.n_clusters,
        'model_roc_auc': MODEL_METRICS.get('roc_auc', '—'),
        'model_accuracy': MODEL_METRICS.get('accuracy', '—'),
        'model_mae': MODEL_METRICS.get('mae', '—'),
        'error': None,
        'segment': None,
        'cluster': None,
        'churn': None,
        'proba': 0,
        'panier': 0,
        'risk_level': "Inconnu",
        'risk_color': "muted",
        'churn_status': "Inconnu",
        'recommendations': [],
    }

    if request.method == "POST":
        template_globals['form_data'] = request.form
        try:
            # Récupération des données
            freq = float(request.form.get("Frequency", 0))
            monetary = float(request.form.get("MonetaryTotal", 0))
            tenure = float(request.form.get("CustomerTenureDays", 0))
            avg_days = float(request.form.get("AvgDaysBetweenPurchases", 0))
            total_trans = float(request.form.get("TotalTransactions", 0))
            qty = float(request.form.get("TotalQuantity", 0))
            unique = float(request.form.get("UniqueProducts", 0))
            age = float(request.form.get("Age", 0))
            tickets = float(request.form.get("SupportTicketsCount", 0))

            if freq <= 0: raise ValueError("La fréquence doit être > 0")

            # 1. CLUSTERING
            client_cluster = pd.DataFrame([[freq, monetary, tenure, avg_days, total_trans]], columns=cluster_features)
            X_scaled = scaler_cluster.transform(client_cluster)
            X_pca = pca.transform(X_scaled)
            cluster_id = int(kmeans.predict(X_pca)[0])
            segment = interpreter_cluster(cluster_id)

            # 2. FEATURE ENGINEERING (CORRIGÉ : pas de +1)
            avg_basket = monetary / freq

            # 3. CLASSIFICATION CHURN
            input_clf = pd.DataFrame([{
                'Frequency': freq, 'MonetaryTotal': monetary, 'CustomerTenureDays': tenure,
                'AvgDaysBetweenPurchases': avg_days, 'TotalTransactions': total_trans,
                'TotalQuantity': qty, 'UniqueProducts': unique, 'Age': age,
                'SupportTicketsCount': tickets, 'AvgBasketValue': avg_basket, 'CancelRatio': 0.0
            }]).reindex(columns=clf_columns, fill_value=0.0)
            
            X_clf_scaled = scaler_clf.transform(input_clf)
            proba = round(float(clf.predict_proba(X_clf_scaled)[0][1]) * 100, 1)
            churn_pred = 1 if proba >= 50 else 0

            # 4. RÉSULTATS
            risk_level, risk_color, churn_status = get_risk_label(proba)
            
            template_globals.update({
                'segment': segment, 'cluster': cluster_id, 'churn': churn_pred,
                'proba': proba, 'panier': round(avg_basket, 2),
                'risk_level': risk_level, 'risk_color': risk_color,
                'churn_status': churn_status, 'recommendations': get_recommendations(proba, segment)
            })

        except Exception as e:
            template_globals['error'] = str(e)

    return render_template("index.html", **template_globals)

if __name__ == "__main__":
    app.run(debug=True)