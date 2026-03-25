# ============================================================
# src/preprocessing.py
# Nettoyage et préparation des données — VERSION FINALE
# ============================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import afficher_valeurs_manquantes, detecter_outliers_iqr


# ============================================================
# ÉTAPE 1 : Charger les données brutes
# ============================================================

def charger_donnees(chemin='../data/raw/data_original.csv'):
    df = pd.read_csv(chemin)
    print(f"✅ Données chargées : {df.shape[0]} lignes, {df.shape[1]} colonnes")
    return df


# ============================================================
# ÉTAPE 1b : Parser les dates et IP AVANT suppression
# ============================================================

def parser_dates(df):
    """
    Parse RegistrationDate et LastLoginIP AVANT de les supprimer.

    Pourquoi faire ça EN PREMIER ?
    → Si on supprime d'abord, on perd l'information pour toujours !
    → On extrait ce qui est utile, PUIS on supprime la colonne brute

    RegistrationDate → RegYear, RegMonth, RegDay, RegAnciennete
    LastLoginIP      → IP_privee (1=réseau local, 0=internet)
    """
    df = df.copy()

    # ── RegistrationDate ──────────────────────────────────────
    if 'RegistrationDate' in df.columns:

        # Convertir en format date
        # dayfirst=True → priorité format UK (jour/mois/année)
        # errors='coerce' → si format inconnu → NaT (pas de crash)
        df['RegistrationDate'] = pd.to_datetime(
            df['RegistrationDate'],
            format='mixed',    # ← accepte plusieurs formats mélangés
            dayfirst=True,
            errors='coerce'
        )

        # Extraire des features utiles depuis la date
        df['RegYear']  = df['RegistrationDate'].dt.year
        df['RegMonth'] = df['RegistrationDate'].dt.month
        df['RegDay']   = df['RegistrationDate'].dt.day

        # Ancienneté = nombre de jours depuis l'inscription jusqu'à aujourd'hui
        # → Plus ce chiffre est grand, plus le client est ancien
        aujourd_hui = pd.Timestamp.today()
        df['RegAnciennete'] = (aujourd_hui - df['RegistrationDate']).dt.days

        # Remplir les NaT éventuels par la médiane
        for col in ['RegYear', 'RegMonth', 'RegDay', 'RegAnciennete']:
            df[col] = df[col].fillna(df[col].median())

        print("✅ RegistrationDate → RegYear, RegMonth, RegDay, RegAnciennete créées")

    # ── LastLoginIP ───────────────────────────────────────────
    if 'LastLoginIP' in df.columns:

        # Détecter si l'IP est privée (réseau local) ou publique (internet)
        # IP privées connues : 192.168.x.x / 10.x.x.x / 172.16.x.x
        # Un client avec IP privée = connecté depuis le bureau (B2B ?)
        def est_ip_privee(ip):
            if pd.isna(ip):
                return 0
            ip_str = str(ip)
            return int(
                ip_str.startswith('192.168') or
                ip_str.startswith('10.')     or
                ip_str.startswith('172.16')
            )

        df['IP_privee'] = df['LastLoginIP'].apply(est_ip_privee)
        print("✅ LastLoginIP      → IP_privee créée (1=privée, 0=publique)")

    return df


# ============================================================
# ÉTAPE 2 : Supprimer les colonnes inutiles
# ============================================================

def supprimer_colonnes_inutiles(df):
    """
    Supprime les colonnes inutiles ou dangereuses.

    Pourquoi chaque colonne ?
    ─────────────────────────────────────────────────────────
    CustomerID          → identifiant pur, ne prédit rien
    NewsletterSubscribed→ toujours 'Yes' = variance nulle
    LastLoginIP         → déjà transformée en IP_privee
    RegistrationDate    → déjà transformée en RegYear/Month/Day
    ChurnRiskCategory   → DATA LEAKAGE ! calculée à partir de Churn
                          Si on la garde, le modèle "triche" car
                          il voit déjà la réponse avant de prédire
    """
    colonnes_a_supprimer = [
        'CustomerID',
        'NewsletterSubscribed',
        'LastLoginIP',           # remplacée par IP_privee
        'RegistrationDate',      # remplacée par RegYear/Month/Day/Anciennete
        'ChurnRiskCategory'      # DATA LEAKAGE → À SUPPRIMER ABSOLUMENT
    ]

    colonnes_presentes = [col for col in colonnes_a_supprimer if col in df.columns]
    df = df.drop(columns=colonnes_presentes)

    print(f"✅ Colonnes supprimées : {colonnes_presentes}")
    print(f"🔎 Nombre supprimé     : {len(colonnes_presentes)}")
    print(f"🔎 Colonnes restantes  : {df.shape[1]}")
    return df


# ============================================================
# ÉTAPE 3 : Corriger les valeurs aberrantes
# ============================================================

def corriger_valeurs_aberrantes(df):
    """
    Corrige les codes erreurs déguisés en chiffres.

    SupportTicketsCount : -1 et 999 → NaN  (valides : 0-15)
    SatisfactionScore   : -1 et 99  → NaN  (valides : 1-5)
    """
    df = df.copy()

    if 'SupportTicketsCount' in df.columns:
        nb = df['SupportTicketsCount'].isin([-1, 999]).sum()
        df.loc[df['SupportTicketsCount'].isin([-1, 999]), 'SupportTicketsCount'] = np.nan
        print(f"✅ SupportTicketsCount : {nb} valeurs aberrantes → NaN")

    if 'SatisfactionScore' in df.columns:
        nb = df['SatisfactionScore'].isin([-1, 99]).sum()
        df.loc[df['SatisfactionScore'].isin([-1, 99]), 'SatisfactionScore'] = np.nan
        print(f"✅ SatisfactionScore   : {nb} valeurs aberrantes → NaN")

    return df


# ============================================================
# ÉTAPE 4 : Imputer les valeurs manquantes
# ============================================================

def imputer_valeurs_manquantes(df):
    """
    Remplace les NaN par des valeurs estimées.

    Numériques  → médiane (résistante aux outliers)
    Texte       → 'Inconnu'
    """
    df = df.copy()

    colonnes_mediane = ['Age', 'SupportTicketsCount', 'SatisfactionScore']

    for col in colonnes_mediane:
        if col in df.columns:
            nb_nan  = df[col].isnull().sum()
            mediane = df[col].median()
            df[col] = df[col].fillna(mediane)
            print(f"✅ {col} : {nb_nan} NaN → médiane ({mediane:.1f})")

    colonnes_cat = df.select_dtypes(include=['object', 'string']).columns
    for col in colonnes_cat:
        nb_nan = df[col].isnull().sum()
        if nb_nan > 0:
            df[col] = df[col].fillna('Inconnu')
            print(f"✅ {col} : {nb_nan} NaN → 'Inconnu'")

    return df


# ============================================================
# ÉTAPE 5 : Feature Engineering
# ============================================================

def feature_engineering(df):
    """
    Crée de nouvelles colonnes utiles à partir des existantes.
    """
    df = df.copy()

    if 'MonetaryTotal' in df.columns and 'Recency' in df.columns:
        df['MonetaryPerDay'] = df['MonetaryTotal'] / (df['Recency'] + 1)
        print("✅ Feature créée : MonetaryPerDay")

    if 'MonetaryTotal' in df.columns and 'Frequency' in df.columns:
        df['AvgBasketValue'] = df['MonetaryTotal'] / (df['Frequency'] + 1)
        print("✅ Feature créée : AvgBasketValue")

    if 'Recency' in df.columns and 'CustomerTenureDays' in df.columns:
        df['TenureRatio'] = df['Recency'] / (df['CustomerTenureDays'] + 1)
        print("✅ Feature créée : TenureRatio")

    if 'CancelledTransactions' in df.columns and 'TotalTransactions' in df.columns:
        df['CancelRatio'] = df['CancelledTransactions'] / (df['TotalTransactions'] + 1)
        print("✅ Feature créée : CancelRatio")

    return df


# ============================================================
# ÉTAPE 6 : Encoder les colonnes catégorielles
# ============================================================

def encoder_colonnes(df):
    """
    Convertit les colonnes texte en nombres.

    3 méthodes :
    - Ordinal  : colonnes avec ordre logique (Low < Medium < High)
    - Target   : Country → taux de churn moyen par pays (1 colonne)
    - One-Hot  : colonnes sans ordre → colonnes 0/1
    """
    df = df.copy()

    # ── ORDINAL ENCODING ──────────────────────────────────────

    if 'SpendingCategory' in df.columns:
        ordre = ['Low', 'Medium', 'High', 'VIP']
        df['SpendingCategory'] = pd.Categorical(
            df['SpendingCategory'], categories=ordre, ordered=True).codes
        print("✅ SpendingCategory    → ordinal (0 à 3)")

    if 'LoyaltyLevel' in df.columns:
        ordre = ['Inconnu', 'Nouveau', 'Jeune', 'Établi', 'Ancien']
        df['LoyaltyLevel'] = pd.Categorical(
            df['LoyaltyLevel'], categories=ordre, ordered=True).codes
        print("✅ LoyaltyLevel        → ordinal (0 à 4)")

    if 'AgeCategory' in df.columns:
        ordre = ['Inconnu', '18-24', '25-34', '35-44', '45-54', '55-64', '65+']
        df['AgeCategory'] = pd.Categorical(
            df['AgeCategory'], categories=ordre, ordered=True).codes
        print("✅ AgeCategory         → ordinal (0 à 6)")

    if 'BasketSizeCategory' in df.columns:
        ordre = ['Inconnu', 'Petit', 'Moyen', 'Grand']
        df['BasketSizeCategory'] = pd.Categorical(
            df['BasketSizeCategory'], categories=ordre, ordered=True).codes
        print("✅ BasketSizeCategory  → ordinal (0 à 3)")

    if 'PreferredTimeOfDay' in df.columns:
        ordre = ['Nuit', 'Matin', 'Midi', 'Après-midi', 'Soir']
        df['PreferredTimeOfDay'] = pd.Categorical(
            df['PreferredTimeOfDay'], categories=ordre, ordered=True).codes
        print("✅ PreferredTimeOfDay  → ordinal (0 à 4)")

    # ── TARGET ENCODING pour Country ──────────────────────────
    # 37+ pays → 1 seule colonne numérique (taux churn moyen)

    if 'Country' in df.columns:
        df = df.drop(columns=['Country'])
        print("✅ Country supprimée (Target Encoding fait dans train_model.py)")

    # ── ONE-HOT ENCODING ──────────────────────────────────────
    # dtype=int → 0 et 1 au lieu de True/False

    colonnes_onehot = [
        'CustomerType',
        'FavoriteSeason',
        'Region',
        'WeekendPreference',
        'ProductDiversity',
        'Gender',
        'AccountStatus',
        'RFMSegment'
    ]

    colonnes_presentes = [col for col in colonnes_onehot if col in df.columns]

    if colonnes_presentes:
        df = pd.get_dummies(df, columns=colonnes_presentes,
                            drop_first=False, dtype=int)
        print(f"✅ One-Hot Encoding    → 0 et 1 (pas True/False)")
        print(f"   Dimensions après encodage : {df.shape}")

    return df


# ============================================================
# ÉTAPE 7 : Normalisation (appelée APRÈS le split train/test)
# ============================================================

def normaliser(X_train, X_test):
    """
    Centre et réduit les features numériques (moyenne=0, écart-type=1).

    ⚠️ RÈGLE ANTI DATA LEAKAGE :
    fit_transform → sur X_train UNIQUEMENT
    transform     → sur X_test  UNIQUEMENT
    """
    scaler = StandardScaler()
    colonnes_num = X_train.select_dtypes(include=[np.number]).columns.tolist()

    X_train = X_train.copy()
    X_test  = X_test.copy()

    X_train[colonnes_num] = scaler.fit_transform(X_train[colonnes_num])
    X_test[colonnes_num]  = scaler.transform(X_test[colonnes_num])

    print(f"✅ Normalisation sur {len(colonnes_num)} colonnes numériques")
    return X_train, X_test, scaler


# ============================================================
# PIPELINE PRINCIPALE
# ============================================================

def pipeline_preprocessing(
        chemin_input='../data/raw/data_original.csv',
        chemin_output='../data/processed/data_clean.csv'):

    print("\n" + "=" * 55)
    print("DÉMARRAGE DU PREPROCESSING")
    print("=" * 55 + "\n")

    df = charger_donnees(chemin_input)
    # DEBUG colonnes
    nb_initial = df.shape[1]
    print(f"🔎 Colonnes initiales : {nb_initial}")
    print("\n--- Étape 1b : Parsing RegistrationDate + LastLoginIP ---")
    df = parser_dates(df)
    print(f"🔎 Colonnes après parsing (feature engineering initial) : {df.shape[1]}")
    print("\n--- Étape 2 : Suppression des colonnes inutiles ---")
    df = supprimer_colonnes_inutiles(df)

    print("\n--- Étape 3 : Correction des valeurs aberrantes ---")
    df = corriger_valeurs_aberrantes(df)

    print("\n--- Étape 4 : Imputation des valeurs manquantes ---")
    df = imputer_valeurs_manquantes(df)

    print("\n--- Étape 5 : Feature Engineering ---")
    df = feature_engineering(df)
    print(f"🔎 Colonnes après feature engineering : {df.shape[1]}")
    print("\n--- Étape 6 : Encodage ---")
    df = encoder_colonnes(df)

    os.makedirs(os.path.dirname(chemin_output), exist_ok=True)
    df.to_csv(chemin_output, index=False)
    print(f"\n✅ Fichier propre sauvegardé : {chemin_output}")
    print(f"   Dimensions finales : {df.shape[0]} lignes x {df.shape[1]} colonnes")

    print("\n--- Étape 7 : Split Train/Test (80/20 stratifié) ---")
    from sklearn.model_selection import train_test_split

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train = X_train.reset_index(drop=True)
    X_test  = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test  = y_test.reset_index(drop=True)

    os.makedirs("../data/train_test", exist_ok=True)
    X_train.to_csv("../data/train_test/X_train.csv", index=False)
    X_test.to_csv("../data/train_test/X_test.csv",   index=False)
    y_train.to_csv("../data/train_test/y_train.csv", index=False)
    y_test.to_csv("../data/train_test/y_test.csv",   index=False)

    train = pd.concat([X_train, y_train], axis=1)
    test  = pd.concat([X_test,  y_test],  axis=1)
    train.to_csv("../data/train_test/train.csv", index=False)
    test.to_csv("../data/train_test/test.csv",   index=False)

    print(f"✅ Train : {X_train.shape[0]} lignes | Test : {X_test.shape[0]} lignes")
    print("✅ Split train/test sauvegarde dans data/train_test/")
    print("\n" + "=" * 55)
    print("PREPROCESSING TERMINE")
    print("=" * 55)

    return df


if __name__ == "__main__":
    df_propre = pipeline_preprocessing()