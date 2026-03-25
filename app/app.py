# ============================================================
# app/app.py
# Interface Flask — Prédiction client en temps réel
# ============================================================
# USAGE : cd app && python app.py
# Puis ouvrir : http://127.0.0.1:5000
# ============================================================

from flask import Flask, render_template, request
import pandas as pd
import joblib
import os

app = Flask(__name__)

# ============================================================
# CHARGER LES MODÈLES AU DÉMARRAGE
# ============================================================

CHEMIN_MODELS = '../models'

try:
    # Clustering
    kmeans           = joblib.load(os.path.join(CHEMIN_MODELS, 'kmeans.pkl'))
    pca              = joblib.load(os.path.join(CHEMIN_MODELS, 'pca.pkl'))
    scaler_cluster   = joblib.load(os.path.join(CHEMIN_MODELS, 'scaler_cluster.pkl'))
    cluster_features = joblib.load(os.path.join(CHEMIN_MODELS, 'cluster_features.pkl'))

    # Classification
    clf              = joblib.load(os.path.join(CHEMIN_MODELS, 'churn_model.pkl'))
    scaler_clf       = joblib.load(os.path.join(CHEMIN_MODELS, 'scaler_clf.pkl'))
    clf_columns      = joblib.load(os.path.join(CHEMIN_MODELS, 'churn_columns.pkl'))

    # Régression
    reg              = joblib.load(os.path.join(CHEMIN_MODELS, 'regression_model.pkl'))
    scaler_reg       = joblib.load(os.path.join(CHEMIN_MODELS, 'scaler_reg.pkl'))
    reg_columns      = joblib.load(os.path.join(CHEMIN_MODELS, 'reg_columns.pkl'))

    print("✅ Tous les modèles chargés avec succès")

except FileNotFoundError as e:
    print(f"❌ Erreur : modèle introuvable — {e}")
    print("   Lance d'abord : python src/train_model.py")


# ============================================================
# INTERPRÉTATION DES CLUSTERS
# ============================================================

def interpreter_cluster(cluster_id):
    return {
        0: "Clients occasionnels",
        1: "Clients à risque",
        2: "Clients VIP",
        3: "Clients occasionnels",
        4: "Gros acheteurs",
        5: "Clients peu actifs",
        6: "Clients fidèles",
    }.get(int(cluster_id), "Segment inconnu")


# ============================================================
# ROUTE PRINCIPALE
# ============================================================

@app.route("/", methods=["GET", "POST"])
def index():

    if request.method == "POST":
        try:
            # ── Récupérer les inputs du formulaire ──────────
            data = {
                "Recency"                : float(request.form["Recency"]),
                "Frequency"              : float(request.form["Frequency"]),
                "MonetaryTotal"          : float(request.form["MonetaryTotal"]),
                "CustomerTenureDays"     : float(request.form["CustomerTenure"]),
                "AvgDaysBetweenPurchases": float(request.form["AvgDaysBetween"]),
                "TotalTransactions"      : float(request.form["TotalTrans"]),
            }
            df = pd.DataFrame([data])

            # ── Clustering ───────────────────────────────────
            df_c       = df.reindex(columns=cluster_features, fill_value=0).astype(float)
            X_sc       = scaler_cluster.transform(df_c)
            X_pca      = pca.transform(X_sc)
            cluster    = int(kmeans.predict(X_pca)[0])
            segment    = interpreter_cluster(cluster)

            # ── Classification (Churn) ────────────────────────
            df_clf_in  = pd.get_dummies(df.copy())
            df_clf_in  = df_clf_in.reindex(columns=clf_columns, fill_value=0)
            X_clf_sc   = scaler_clf.transform(df_clf_in)
            churn      = int(clf.predict(X_clf_sc)[0])
            proba      = round(float(clf.predict_proba(X_clf_sc)[0][1]), 2)

            # ── Régression (Revenu) ───────────────────────────
            df_reg_in  = pd.get_dummies(df.copy())
            df_reg_in  = df_reg_in.reindex(columns=reg_columns, fill_value=0)
            X_reg_sc   = scaler_reg.transform(df_reg_in)
            revenue    = round(float(reg.predict(X_reg_sc)[0]), 2)

            return render_template(
                "index.html",
                segment       = segment,
                cluster       = cluster,
                churn         = churn,
                proba         = proba,
                revenue       = revenue,
                error         = None
            )

        except Exception as e:
            return render_template("index.html", error=str(e))

    return render_template("index.html")


# ============================================================
# LANCEMENT
# ============================================================

if __name__ == "__main__":
    app.run(debug=True)