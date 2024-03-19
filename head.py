import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def main():
    # Charger le fichier JSON dans un DataFrame
    try:
        df = pd.read_json('orientation_esgis_base2.json')
    except FileNotFoundError:
        print("Error: File 'orientation_esgis_base2.json' not found.")
        return
    except Exception as e:
        print("Error loading JSON file:", str(e))
        return

    # Afficher les premières lignes du DataFrame
    print("Premières lignes du DataFrame :")
    print(df.head())
    print("\n")

    # Afficher les dernières lignes du DataFrame
    print("Dernières lignes du DataFrame :")
    print(df.tail())
    print("\n")

    if 'patterns' not in df.columns:
        print("Error: 'patterns' column not found in the DataFrame.")
        return

    if 'responses' not in df.columns:
        print("Error: 'responses' column not found in the DataFrame.")
        return

    # Calculer la longueur des patterns et des réponses
    df['pattern_length'] = df['patterns'].apply(lambda x: len(x))
    df['response_length'] = df['responses'].apply(lambda x: len(x))

    # Afficher les statistiques descriptives
    print("Statistiques descriptives des longueurs de patterns et de réponses :")
    print(df[['pattern_length', 'response_length']].describe())

    # Créer un heatmap de corrélation
    corr = df.corr()
    sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, cmap='coolwarm', annot=True)
    plt.title('Heatmap de Corrélation')
    plt.show()

if __name__ == "__main__":
    main()