import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def main():
    # Charger le fichier JSON dans un DataFrame
    try:
        df = pd.read_json('orientation_esgis_base2.json')
    except FileNotFoundError:
        print("Erreur : fichier 'orientation_esgis_base2.json' introuvable.")
        return
    except Exception as e:
        print("Erreur lors du chargement du fichier JSON :", str(e))
        return

    # Afficher les premières lignes du DataFrame
    print("Premières lignes du DataFrame :")
    print(df.head().to_string(index=False))
    print("\n")

    # Afficher les dernières lignes du DataFrame
    print("Dernières lignes du DataFrame :")
    print(df.tail().to_string(index=False))
    print("\n")

    # Vérifier la présence des colonnes 'patterns' et 'responses'
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
    print(df[['pattern_length', 'response_length']].describe().to_string())
    print("\n")

    # Créer un heatmap de corrélation
    corr = df.corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, cmap='coolwarm', annot=True)
    plt.title('Heatmap de Corrélation')
    plt.show()

    # Effectuer des analyses statistiques avec la fonction ensemble
    print("Analyses statistiques avec la fonction ensemble :")
    ensemble_stats = df[['pattern_length', 'response_length']].agg(['mean', 'std', 'min', 'max']).transpose()
    print(ensemble_stats)

if __name__ == "__main__":
    main()
