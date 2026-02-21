# ============================================================
# src/utils.py
# Fonctions utilitaires réutilisables dans tout le projet
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


def afficher_valeurs_manquantes(df):
    """
    Affiche un résumé des valeurs manquantes dans le dataframe.
    
    Paramètre : df = votre dataframe pandas
    Retourne   : un dataframe trié par % manquant
    """
    valeurs_manquantes = df.isnull().sum()
    pourcentage = (valeurs_manquantes / len(df)) * 100
    
    resume = pd.DataFrame({
        'Valeurs manquantes': valeurs_manquantes,
        'Pourcentage (%)': pourcentage.round(2)
    })
    
    # Garder seulement les colonnes avec des NaN
    resume = resume[resume['Valeurs manquantes'] > 0]
    resume = resume.sort_values('Pourcentage (%)', ascending=False)
    
    return resume


def detecter_outliers_iqr(df, colonne):
    """
    Détecte les valeurs aberrantes avec la méthode IQR (écart interquartile).
    
    Explication simple :
    - Q1 = 25% des données sont en dessous
    - Q3 = 75% des données sont en dessous
    - IQR = Q3 - Q1 (l'écart entre les deux)
    - Outlier = toute valeur en dehors de [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
    
    Paramètres : df = dataframe, colonne = nom de la colonne à analyser
    Retourne   : le nombre d'outliers et les bornes
    """
    Q1 = df[colonne].quantile(0.25)
    Q3 = df[colonne].quantile(0.75)
    IQR = Q3 - Q1
    
    borne_basse = Q1 - 1.5 * IQR
    borne_haute = Q3 + 1.5 * IQR
    
    outliers = df[(df[colonne] < borne_basse) | (df[colonne] > borne_haute)]
    
    print(f"[{colonne}]")
    print(f"  Borne basse  : {borne_basse:.2f}")
    print(f"  Borne haute  : {borne_haute:.2f}")
    print(f"  Nb outliers  : {len(outliers)} ({len(outliers)/len(df)*100:.2f}%)")
    
    return len(outliers), borne_basse, borne_haute


def sauvegarder_graphique(nom_fichier, dossier='../reports'):
    """
    Sauvegarde le graphique matplotlib actuel dans le dossier reports/.
    
    Paramètres :
    - nom_fichier = ex: 'distribution_age.png'
    - dossier     = chemin du dossier de sauvegarde
    """
    os.makedirs(dossier, exist_ok=True)  # Crée le dossier si inexistant
    chemin = os.path.join(dossier, nom_fichier)
    plt.savefig(chemin, bbox_inches='tight', dpi=150)
    print(f"✅ Graphique sauvegardé : {chemin}")


def afficher_distribution(df, colonne, titre=None):
    """
    Affiche l'histogramme + boxplot d'une colonne numérique.
    
    Paramètres : df = dataframe, colonne = nom de la colonne
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Histogramme
    df[colonne].dropna().hist(bins=30, ax=ax1, color='steelblue', edgecolor='white')
    ax1.set_title(f'Distribution de {colonne}')
    ax1.set_xlabel(colonne)
    ax1.set_ylabel('Fréquence')
    
    # Boxplot
    ax2.boxplot(df[colonne].dropna())
    ax2.set_title(f'Boxplot de {colonne}')
    ax2.set_ylabel(colonne)
    
    if titre:
        fig.suptitle(titre, fontsize=13)
    
    plt.tight_layout()
    plt.show()


def resume_dataset(df):
    """
    Affiche un résumé complet du dataset en une seule fonction.
    
    Paramètre : df = votre dataframe
    """
    print("=" * 50)
    print("📊 RÉSUMÉ DU DATASET")
    print("=" * 50)
    print(f"  Lignes        : {df.shape[0]:,}")
    print(f"  Colonnes      : {df.shape[1]}")
    print(f"  Doublons      : {df.duplicated().sum()}")
    print(f"  Mémoire       : {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print()
    
    # Types de colonnes
    types = df.dtypes.value_counts()
    print("📋 Types de colonnes :")
    for t, count in types.items():
        print(f"   {t} : {count} colonnes")
    
    print()
    
    # Valeurs manquantes
    nb_nan = df.isnull().sum().sum()
    print(f"❓ Total valeurs manquantes : {nb_nan:,}")
    
    print("=" * 50)

def afficher_correlation(df, seuil=0.8):
    """
    Affiche les paires de colonnes trop corrélées (multicolinéarité).
    Seuil recommandé par votre document : 0.8
    """
    df_num = df.select_dtypes(include=[np.number])
    matrice = df_num.corr().abs()

    paires = []
    cols = matrice.columns
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            if matrice.iloc[i, j] >= seuil:
                paires.append({
                    'Feature 1'   : cols[i],
                    'Feature 2'   : cols[j],
                    'Corrélation' : round(matrice.iloc[i, j], 3)
                })

    if paires:
        result = pd.DataFrame(paires).sort_values('Corrélation', ascending=False)
        print(f"⚠️  {len(paires)} paires avec corrélation ≥ {seuil} :")
        print(result.to_string(index=False))
    else:
        print(f"✅ Aucune paire avec corrélation ≥ {seuil}")

    return pd.DataFrame(paires)