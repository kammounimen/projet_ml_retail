# ============================================================
# src/preprocessing.py
# Nettoyage et préparation des données — VERSION CORRIGÉE AVEC FILTRAGE
# (Anti Data Leakage + Nettoyage valeurs négatives)
# ============================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ============================================================
# ÉTAPE 1 : Charger les données brutes
# ============================================================

def charger_donnees(chemin='../data/raw/data_original.csv'):
    df = pd.read_csv(chemin)
    print(f"Donnees chargees : {df.shape[0]} lignes, {df.shape[1]} colonnes")
    return df


# ============================================================
# ÉTAPE 1b : Parser les dates et IP AVANT suppression
# ============================================================

def parser_dates(df):
    """
    Parse RegistrationDate et LastLoginIP AVANT de les supprimer.

    CORRECTION ANTI-LEAKAGE :
    → date_reference FIXE au lieu de pd.Timestamp.today()
    → Sinon RegAnciennete change chaque jour = instabilité en production

    Date de référence = 2013-01-07
    (calculée depuis les données : RegistrationDate_max + Recency_max)
    """
    df = df.copy()

    DATE_REFERENCE = pd.Timestamp('2013-01-07')

    if 'RegistrationDate' in df.columns:
        df['RegistrationDate'] = pd.to_datetime(
            df['RegistrationDate'],
            format='mixed',
            dayfirst=True,
            errors='coerce'
        )

        df['RegYear']      = df['RegistrationDate'].dt.year
        df['RegMonth']     = df['RegistrationDate'].dt.month
        df['RegDay']       = df['RegistrationDate'].dt.day
        df['RegAnciennete'] = (DATE_REFERENCE - df['RegistrationDate']).dt.days

        # NOTE : les NaN de ces colonnes seront imputés APRÈS le split
        # (dans imputer_valeurs_manquantes_post_split) pour éviter la fuite
        print("RegistrationDate -> RegYear, RegMonth, RegDay, RegAnciennete")
        print(f"   Date de reference fixe : {DATE_REFERENCE.date()}")

    if 'LastLoginIP' in df.columns:
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
        print("LastLoginIP -> IP_privee (1=privee, 0=publique)")

    return df


# ============================================================
# ÉTAPE 2 : Supprimer les colonnes inutiles + fuites
# ============================================================

def supprimer_colonnes_inutiles(df):
    """
    Supprime les colonnes inutiles ou dangereuses.

    FUITES CONFIRMÉES :
    CustomerType      → valeur "Perdu" = 100% churn=1  (FUITE TOTALE)
    ChurnRiskCategory → valeur "Critique" = 100% churn=1  (FUITE TOTALE)
    AccountStatus     → 'Closed' très corrélé au churn  (FUITE FORTE)
    LoyaltyLevel      → clients churnés jamais 'Établi'/'Ancien' (FUITE FORTE)
    RFMSegment        → calculé à partir de Recency (déjà supprimé)

    FUITES INDIRECTES / REDONDANCES :
    SatisfactionScore → score collecté après le churn
    AgeCategory       → doublon de Age
    BasketSizeCategory→ doublon de AvgLinesPerInvoice
    SpendingCategory  → doublon catégoriel de MonetaryTotal

    COLONNES BRUTES (déjà transformées) :
    CustomerID        → identifiant pur, ne prédit rien
    NewsletterSubscribed → variance nulle (toujours 'Yes')
    LastLoginIP       → transformée en IP_privee
    RegistrationDate  → transformée en RegYear/Month/Day/Anciennete
    Recency           → utilisé pour calculer le Churn
    """
    colonnes_a_supprimer = [
        'CustomerID',
        'NewsletterSubscribed',
        'LastLoginIP',
        'RegistrationDate',

        # FUITES TOTALES
        'CustomerType',
        'ChurnRiskCategory',
        'AccountStatus',

        # FUITES FORTES
        'LoyaltyLevel',
        'Recency',
        'RFMSegment',

        # FUITES INDIRECTES / REDONDANCES
        'SatisfactionScore',
        'AgeCategory',
        'BasketSizeCategory',
        'SpendingCategory',

        # DOUBLONS PARFAITS (correlation = 1.0 confirmee sur les donnees)
        'NegativeQuantityCount',
        'UniqueInvoices',
        'UniqueDescriptions',
        'ReturnRatio',
        'PreferredHour',
        'AvgProductsPerTransaction',

        # FUITES POTENTIELLES TEMPORELLES / COMPORTEMENTALES
        'FirstPurchaseDaysAgo',
        'PreferredMonth',

        # FUITES TEMPORELLES INDIRECTES (confirmees par analyse des donnees)
        # FavoriteSeason : calculee sur historique incluant la periode de churn
        # -> corr avec RegYear : clients recents ont massivement 'Automne'
        # -> le modele utilise la saison pour deviner l'anciennete -> fuite
        'FavoriteSeason',
        'RegYear',
        'RegMonth',
        'RegDay',
        'PreferredDayOfWeek',
        'WeekendPurchaseRatio',
    ]

    colonnes_presentes = [col for col in colonnes_a_supprimer if col in df.columns]
    df = df.drop(columns=colonnes_presentes)

    print(f"Colonnes supprimees : {colonnes_presentes}")
    print(f"Nombre supprime     : {len(colonnes_presentes)}")
    print(f"Colonnes restantes  : {df.shape[1]}")
    return df


# ============================================================
# ÉTAPE 3 : Corriger les valeurs aberrantes
# ============================================================

def corriger_valeurs_aberrantes(df):
    """
    Corrige les codes erreurs déguisés en chiffres.
    SupportTicketsCount : -1 et 999 → NaN  (valides : 0-15)
    """
    df = df.copy()

    if 'SupportTicketsCount' in df.columns:
        nb = df['SupportTicketsCount'].isin([-1, 999]).sum()
        df.loc[df['SupportTicketsCount'].isin([-1, 999]), 'SupportTicketsCount'] = np.nan
        print(f"SupportTicketsCount : {nb} valeurs aberrantes -> NaN")

    return df


# ============================================================
# ÉTAPE 3b : Filtrer les valeurs négatives (NOUVEAU)
# ============================================================

def filtrer_valeurs_negatives(df):
    """
    Filtre les valeurs impossibles comme MonetaryTotal négatif.
    Ces valeurs viennent probablement d'annulations ou d'erreurs.
    """
    df = df.copy()
    avant = len(df)
    
    if 'MonetaryTotal' in df.columns:
        df = df[df['MonetaryTotal'] > 0]
        print(f"MonetaryTotal <= 0 : {avant - len(df)} lignes supprimées")
        avant = len(df)
    
    if 'Frequency' in df.columns:
        df = df[df['Frequency'] >= 1]
        print(f"Frequency < 1 : {avant - len(df)} lignes supprimées")
    
    if 'CustomerTenureDays' in df.columns:
        df = df[df['CustomerTenureDays'] >= 0]
        print(f"CustomerTenureDays < 0 : {avant - len(df)} lignes supprimées")
    
    return df


# ============================================================
# ÉTAPE 4 : Feature Engineering
# ============================================================

def feature_engineering(df):
    """
    Crée de nouvelles colonnes utiles à partir des existantes.

    NOTE : MonetaryPerDay et TenureRatio supprimés car ils
    utilisaient Recency (déjà supprimée pour cause de fuite).
    """
    df = df.copy()

    if 'MonetaryTotal' in df.columns and 'Frequency' in df.columns:
        df['AvgBasketValue'] = df['MonetaryTotal'] / (df['Frequency'] + 1)
        print("Feature creee : AvgBasketValue")

    if 'CancelledTransactions' in df.columns and 'TotalTransactions' in df.columns:
        df['CancelRatio'] = df['CancelledTransactions'] / (df['TotalTransactions'] + 1)
        print("Feature creee : CancelRatio")

    return df


# ============================================================
# ÉTAPE 5 : Encoder les colonnes catégorielles
# ============================================================

def encoder_colonnes(df):
    """
    Convertit les colonnes texte en nombres.
    """
    df = df.copy()

    if 'PreferredTimeOfDay' in df.columns:
        ordre = ['Nuit', 'Matin', 'Midi', 'Après-midi', 'Soir']
        df['PreferredTimeOfDay'] = pd.Categorical(
            df['PreferredTimeOfDay'], categories=ordre, ordered=True).codes
        print("PreferredTimeOfDay -> ordinal (0 a 4)")

    if 'Country' in df.columns:
        df = df.drop(columns=['Country'])
        print("Country supprimee (Target Encoding a faire dans train_model.py)")

    colonnes_onehot = [
        'FavoriteSeason',
        'Region',
        'WeekendPreference',
        'ProductDiversity',
        'Gender'
    ]

    colonnes_presentes = [col for col in colonnes_onehot if col in df.columns]

    if colonnes_presentes:
        df = pd.get_dummies(df, columns=colonnes_presentes,
                            drop_first=False, dtype=int)
        print(f"One-Hot Encoding -> entiers 0/1")
        print(f"Dimensions apres encodage : {df.shape}")

    return df


# ============================================================
# ÉTAPE 6 : Imputation POST-SPLIT (CORRIGÉE — anti data leakage)
# ============================================================

def imputer_valeurs_manquantes_post_split(X_train, X_test):
    """
    Impute les valeurs manquantes APRÈS le split train/test.

    RÈGLE ANTI-LEAKAGE :
    - La médiane est calculée UNIQUEMENT sur X_train
    - Elle est ensuite appliquée à X_train ET X_test
    - Cela évite que X_test influence les statistiques d'imputation

    AVANT (version incorrecte) : imputation sur tout le dataset → fuite
    APRÈS (version corrigée)   : imputation fit sur X_train uniquement
    """
    X_train = X_train.copy()
    X_test  = X_test.copy()

    colonnes_mediane = ['Age', 'SupportTicketsCount']

    for col in colonnes_mediane:
        if col in X_train.columns:
            # Médiane calculée UNIQUEMENT sur X_train
            mediane_train = X_train[col].median()
            nb_nan_train  = X_train[col].isnull().sum()
            nb_nan_test   = X_test[col].isnull().sum()

            X_train[col] = X_train[col].fillna(mediane_train)
            X_test[col]  = X_test[col].fillna(mediane_train)  # même médiane du train

            print(f"{col} : train={nb_nan_train} NaN, test={nb_nan_test} NaN "
                  f"-> mediane train ({mediane_train:.1f})")

    # Colonnes RegAnciennete, RegYear, RegMonth, RegDay : même logique
    colonnes_dates = ['RegAnciennete', 'RegYear', 'RegMonth', 'RegDay']
    for col in colonnes_dates:
        if col in X_train.columns and X_train[col].isnull().any():
            mediane_train = X_train[col].median()
            X_train[col] = X_train[col].fillna(mediane_train)
            X_test[col]  = X_test[col].fillna(mediane_train)
            print(f"{col} : NaN -> mediane train ({mediane_train:.1f})")

    # Colonnes catégorielles restantes → 'Inconnu'
    colonnes_cat_train = X_train.select_dtypes(include=['object', 'string']).columns
    for col in colonnes_cat_train:
        nb_nan = X_train[col].isnull().sum()
        if nb_nan > 0:
            X_train[col] = X_train[col].fillna('Inconnu')
            print(f"{col} (train) : {nb_nan} NaN -> 'Inconnu'")

    colonnes_cat_test = X_test.select_dtypes(include=['object', 'string']).columns
    for col in colonnes_cat_test:
        nb_nan = X_test[col].isnull().sum()
        if nb_nan > 0:
            X_test[col] = X_test[col].fillna('Inconnu')
            print(f"{col} (test) : {nb_nan} NaN -> 'Inconnu'")

    return X_train, X_test


# ============================================================
# ÉTAPE 7 : Normalisation (appelée APRÈS le split train/test)
# ============================================================

def normaliser(X_train, X_test):
    """
    Centre et reduit les features numeriques continues uniquement.

    REGLE ANTI DATA LEAKAGE :
    fit_transform sur X_train UNIQUEMENT
    transform     sur X_test  UNIQUEMENT

    IMPORTANT : les colonnes binaires (One-Hot 0/1) sont exclues
    de la normalisation — les normaliser cree des valeurs decimales
    parasites que le modele exploite comme signal fantome.
    """
    scaler = StandardScaler()

    X_train = X_train.copy()
    X_test  = X_test.copy()

    # Identifier les colonnes binaires (valeurs uniquement 0 et 1)
    def est_binaire(col):
        valeurs = set(X_train[col].dropna().unique())
        return valeurs.issubset({0, 1, 0.0, 1.0})

    toutes_num = X_train.select_dtypes(include=[np.number]).columns.tolist()
    colonnes_binaires = [c for c in toutes_num if est_binaire(c)]
    colonnes_continues = [c for c in toutes_num if c not in colonnes_binaires]

    X_train[colonnes_continues] = scaler.fit_transform(X_train[colonnes_continues])
    X_test[colonnes_continues]  = scaler.transform(X_test[colonnes_continues])

    print(f"Normalisation sur {len(colonnes_continues)} colonnes continues")
    print(f"Colonnes binaires exclues (One-Hot) : {len(colonnes_binaires)}")
    return X_train, X_test, scaler


# ============================================================
# PIPELINE PRINCIPALE — ORDRE CORRIGÉ
# ============================================================

def pipeline_preprocessing(
        chemin_input='../data/raw/data_original.csv',
        chemin_output='../data/processed/data_clean.csv'):

    print("\n" + "=" * 55)
    print("DEMARRAGE DU PREPROCESSING")
    print("=" * 55 + "\n")

    # --- Étapes AVANT le split (transformations sur tout le dataset) ---

    df = charger_donnees(chemin_input)
    print(f"Colonnes initiales : {df.shape[1]}")

    print("\n--- Etape 1b : Parsing RegistrationDate + LastLoginIP ---")
    df = parser_dates(df)
    print(f"Colonnes apres parsing : {df.shape[1]}")

    print("\n--- Etape 2 : Suppression des colonnes inutiles / fuites ---")
    df = supprimer_colonnes_inutiles(df)

    print("\n--- Etape 3 : Correction des valeurs aberrantes ---")
    df = corriger_valeurs_aberrantes(df)

    print("\n--- Etape 3b : Filtrage des valeurs negatives ---")
    df = filtrer_valeurs_negatives(df)

    # NOTE : L'imputation N'EST PLUS faite ici pour éviter la fuite de données.
    # Elle sera faite APRÈS le split, séparément sur X_train et X_test.

    print("\n--- Etape 4 : Feature Engineering ---")
    df = feature_engineering(df)
    print(f"Colonnes apres feature engineering : {df.shape[1]}")

    print("\n--- Etape 5 : Encodage ---")
    df = encoder_colonnes(df)

    os.makedirs(os.path.dirname(chemin_output), exist_ok=True)
    df.to_csv(chemin_output, index=False)
    print(f"\nFichier propre sauvegarde : {chemin_output}")
    print(f"Dimensions : {df.shape[0]} lignes x {df.shape[1]} colonnes")

    # --- SPLIT TRAIN / TEST (doit être fait AVANT imputation et normalisation) ---

    print("\n--- Etape 6 : Split Train/Test (80/20 stratifie) ---")
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

    print(f"Train : {X_train.shape[0]} lignes | Test : {X_test.shape[0]} lignes")

    # --- Étapes POST-SPLIT (fit sur X_train, apply sur X_train + X_test) ---

    print("\n--- Etape 7 : Imputation post-split (anti data leakage) ---")
    # CORRECTION PRINCIPALE : médiane calculée uniquement sur X_train
    X_train, X_test = imputer_valeurs_manquantes_post_split(X_train, X_test)

    # ✅ CORRECTION CRITIQUE : sauvegarder train.csv AVANT normalisation
    # train_model.py a ses propres scalers (scaler_cluster, scaler_clf).
    # Si train.csv contient deja des z-scores, ces scalers normalisent une 2eme fois.
    # En production app.py envoie de vraies valeurs (Frequency=8, Monetary=500)
    # mais le scaler attendrait des z-scores -> predictions completement fausses.
    os.makedirs("../data/train_test", exist_ok=True)

    train_brut = pd.concat([X_train, y_train], axis=1)
    test_brut  = pd.concat([X_test,  y_test],  axis=1)
    train_brut.to_csv("../data/train_test/train.csv", index=False)
    test_brut.to_csv("../data/train_test/test.csv",   index=False)
    print("Saved train.csv / test.csv (valeurs BRUTES, avant normalisation)")

    print("--- Etape 8 : Normalisation post-split ---")
    X_train_norm, X_test_norm, scaler = normaliser(X_train, X_test)

    X_train_norm.to_csv("../data/train_test/X_train_norm.csv", index=False)
    X_test_norm.to_csv("../data/train_test/X_test_norm.csv",   index=False)
    y_train.to_csv("../data/train_test/y_train.csv", index=False)
    y_test.to_csv("../data/train_test/y_test.csv",   index=False)
    print("Saved X_train_norm.csv / X_test_norm.csv (normalises, pour reference)")

    print(f"Split sauvegarde dans data/train_test/")
    print("\n" + "=" * 55)
    print("PREPROCESSING TERMINE")
    print("=" * 55)

    return df, X_train, X_test, y_train, y_test, scaler


if __name__ == "__main__":
    pipeline_preprocessing()