# ============================================================
# app/app.py
# Interface Flask — Prédiction client en temps réel
# VERSION CORRIGÉE — Labels depuis cluster_labels.py
# ============================================================

from flask import Flask, render_template, request
import pandas as pd
import joblib
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from cluster_labels import interpreter_cluster

app = Flask(__name__)

CHEMIN_MODELS = '../models'

try:
    kmeans           = joblib.load(os.path.join(CHEMIN_MODELS, 'kmeans.pkl'))
    pca              = joblib.load(os.path.join(CHEMIN_MODELS, 'pca.pkl'))
    scaler_cluster   = joblib.load(os.path.join(CHEMIN_MODELS, 'scaler_cluster.pkl'))
    cluster_features = joblib.load(os.path.join(CHEMIN_MODELS, 'cluster_features.pkl'))

    clf              = joblib.load(os.path.join(CHEMIN_MODELS, 'churn_model.pkl'))
    scaler_clf       = joblib.load(os.path.join(CHEMIN_MODELS, 'scaler_clf.pkl'))
    clf_columns      = joblib.load(os.path.join(CHEMIN_MODELS, 'churn_columns.pkl'))

    reg              = joblib.load(os.path.join(CHEMIN_MODELS, 'regression_model.pkl'))
    scaler_reg       = joblib.load(os.path.join(CHEMIN_MODELS, 'scaler_reg.pkl'))
    reg_columns      = joblib.load(os.path.join(CHEMIN_MODELS, 'reg_columns.pkl'))

    print("✅ Tous les modèles chargés avec succès")

except FileNotFoundError as e:
    print(f"❌ Erreur : modèle introuvable — {e}")
    print("   Lance d'abord : python src/train_model.py")


@app.route("/", methods=["GET", "POST"])
def index():

    if request.method == "POST":
        try:
            recency     = float(request.form["Recency"])
            frequency   = float(request.form["Frequency"])
            monetary    = float(request.form["MonetaryTotal"])
            tenure      = float(request.form["CustomerTenure"])
            avg_days    = float(request.form["AvgDaysBetween"])
            total_trans = float(request.form["TotalTrans"])

            monetary_per_day = monetary / (recency + 1)
            avg_basket_value = monetary / (frequency + 1)
            tenure_ratio     = recency  / (tenure + 1)

            df = pd.DataFrame([{
                'Recency'                 : recency,
                'Frequency'              : frequency,
                'MonetaryTotal'          : monetary,
                'CustomerTenureDays'     : tenure,
                'AvgDaysBetweenPurchases': avg_days,
                'TotalTransactions'      : total_trans,
            }])

            df_c    = df.reindex(columns=cluster_features, fill_value=0).astype(float)
            X_sc    = scaler_cluster.transform(df_c)
            X_pca   = pca.transform(X_sc)
            cluster = int(kmeans.predict(X_pca)[0])
            segment = interpreter_cluster(cluster)

            df_clf_in = df.copy()
            df_clf_in['MonetaryPerDay'] = monetary_per_day
            df_clf_in['AvgBasketValue'] = avg_basket_value
            df_clf_in['TenureRatio']    = tenure_ratio
            df_clf_in = pd.get_dummies(df_clf_in)
            df_clf_in = df_clf_in.reindex(columns=clf_columns, fill_value=0)
            X_clf_sc  = scaler_clf.transform(df_clf_in)
            churn     = int(clf.predict(X_clf_sc)[0])
            proba     = round(float(clf.predict_proba(X_clf_sc)[0][1]), 2)

            df_reg_in = df.copy()
            df_reg_in['MonetaryPerDay'] = monetary_per_day
            df_reg_in['AvgBasketValue'] = avg_basket_value
            df_reg_in['TenureRatio']    = tenure_ratio
            df_reg_in = pd.get_dummies(df_reg_in)
            df_reg_in = df_reg_in.reindex(columns=reg_columns, fill_value=0)
            X_reg_sc  = scaler_reg.transform(df_reg_in)
            revenue   = round(float(reg.predict(X_reg_sc)[0]), 2)

            return render_template(
                "index.html",
                segment = segment,
                cluster = cluster,
                churn   = churn,
                proba   = proba,
                revenue = revenue,
                error   = None
            )

        except Exception as e:
            return render_template("index.html", error=str(e))

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)