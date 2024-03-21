import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tkinter as tk
from pandastable import Table
import json
from tabulate import tabulate


def main():
    # Charger le fichier JSON dans un DataFrame
    try:
        with open('orientation_esgis_base2.json') as f:
            data = json.load(f)
            df = pd.DataFrame(data['intents'])
    except FileNotFoundError:
        print("Erreur : fichier 'orientation_esgis_base2.json' introuvable.")
        return
    except Exception as e:
        print("Erreur lors du chargement du fichier JSON :", str(e))
        return

        # Afficher les premières lignes du DataFrame
    print("Premières lignes du DataFrame :")
    print(tabulate(df.head(), headers='keys', tablefmt='grid'))
    print("\n")

    # Afficher les dernières lignes du DataFrame
    print("Dernières lignes du DataFrame :")
    print(tabulate(df.tail(), headers='keys', tablefmt='grid'))
    print("\n")

    if 'patterns' not in df.columns:
        print("Erreur : colonne 'patterns' introuvable dans le DataFrame.")
        return

    if 'responses' not in df.columns:
        print("Erreur : colonne 'responses' introuvable dans le DataFrame.")
        return

    # Calculer la longueur des patterns et des réponses
    df['pattern_length'] = df['patterns'].apply(lambda x: len(x))
    df['response_length'] = df['responses'].apply(lambda x: len(x))

    # Afficher les statistiques descriptives
    print("Statistiques descriptives des longueurs de patterns et de réponses :")
    print(tabulate(df[['pattern_length', 'response_length']].describe(), headers='keys', tablefmt='grid'))
    print("\n")

    # Exclure les colonnes non numériques pour le calcul de corrélation
    numeric_cols = df.select_dtypes(include=['int', 'float']).columns
    corr = df[numeric_cols].corr()

    # Créer un heatmap de corrélation
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, cmap='coolwarm', annot=True)
    plt.title('Heatmap de Corrélation')
    plt.show()

    # Effectuer des analyses statistiques avec la fonction ensemble
    print("Analyses statistiques avec la fonction ensemble :")
    ensemble_stats = df[['pattern_length', 'response_length']].agg(['mean', 'std', 'min', 'max']).transpose()
    print(ensemble_stats.to_string(justify='left', col_space=25))

    display_dataframe(df)


def display_dataframe(df):
    root = tk.Tk()
    root.title("Affichage du DataFrame")

    # Créer un widget Table à partir du DataFrame
    table = Table(root, dataframe=df)
    table.show()

    # Lancer la boucle principale de l'interface graphique
    root.mainloop()


if __name__ == "__main__":
    main()
