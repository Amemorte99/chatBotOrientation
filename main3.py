from __future__ import annotations

import csv
import json

from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

from keras.src.layers import Dropout
from sklearn.preprocessing import LabelEncoder, OneHotEncoder  # Import correct LabelEncoder
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from difflib import get_close_matches
from keras.utils import to_categorical
import pandas as pd
import nltk
import numpy as np

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

stemmer = SnowballStemmer('french')



def load_csv(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path, encoding='utf-8')

def process_input(input_text):
    stop_words = set(stopwords.words('french'))
    tokens = [stemmer.stem(token.lower()) for token in word_tokenize(input_text) if token.lower() not in stop_words]
    return tokens

def load_intents(file_path: str) -> dict:
    with open(file_path, 'r', encoding='utf-8') as file:
        data: dict = json.load(file)
    return data


def save_intents(file_path: str, data: dict):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def process_input(input_text):
    stop_words = set(stopwords.words('french'))
    tokens = [stemmer.stem(token.lower()) for token in word_tokenize(input_text) if token.lower() not in stop_words]
    return tokens


def extract_named_entities(text: str) -> list:
    tagged_tokens = pos_tag(word_tokenize(text))
    named_entities = ne_chunk(tagged_tokens)
    entities = []
    for subtree in named_entities.subtrees():
        if subtree.label() == 'NE':
            entity = ' '.join(word for word, tag in subtree.leaves())
            entities.append(entity)
    return entities


def get_closest_match(user_input: str, patterns: list) -> str | None:
    close_match = get_close_matches(user_input, patterns, n=1, cutoff=0.8)
    if close_match:
        return close_match[0]
    else:
        return None


def find_best_match(user_question: str, patterns: list) -> str | None:
    stop_words = set(stopwords.words('french'))
    user_tokens = [stemmer.stem(token.lower()) for token in word_tokenize(user_question) if
                   token.lower() not in stop_words]
    best_match = None
    best_similarity = 0
    for pattern in patterns:
        pattern_tokens = [stemmer.stem(token.lower()) for token in word_tokenize(pattern) if
                          token.lower() not in stop_words]
        similarity = len(set(user_tokens) & set(pattern_tokens)) / len(set(user_tokens) | set(pattern_tokens))
        if similarity > best_similarity:
            best_similarity = similarity
            best_match = pattern
    if best_similarity >= 0.4:
        return best_match
    else:
        return None


def get_response_for_intent2(intent_tag: str, intents: dict) -> str | None:
    for intent in intents["intents"]:
        if intent["tag"] == intent_tag:
            return np.random.choice(intent["responses"])




def call_second_chatbot():
    # Chargement des données depuis un fichier CSV
    df = pd.read_csv('orientation_esgis_base.csv')

    # Assurez-vous que votre CSV a les colonnes correctes. Ici, nous supposons qu'il y a plusieurs colonnes de motifs.
    # Exemple de noms de colonnes: intents/patterns/0, intents/patterns/1, etc.
    pattern_columns = [col for col in df.columns if 'intents/patterns' in col]
    patterns = df[pattern_columns].apply(lambda row: ' '.join(row.values.astype(str)), axis=1).tolist()
    tags = df['intents/tag'].tolist()  # Changez 'intents/tag' en nom correct de colonne pour les tags si nécessaire

    # Création du tokenizer et conversion des motifs en séquences numériques
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(patterns)
    sequences = tokenizer.texts_to_sequences(patterns)

    # Encodage des tags
    label_encoder = LabelEncoder()
    tags_encoded = label_encoder.fit_transform(tags)
    tags_encoded = to_categorical(tags_encoded)

    # Padding des séquences pour avoir une longueur uniforme
    max_sequence_length = max(len(seq) for seq in sequences)
    X = pad_sequences(sequences, padding='post', maxlen=max_sequence_length)

    # Définition de l'architecture du modèle
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=(X.shape[1],)))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(tags_encoded.shape[1], activation='softmax'))

    # Compilation du modèle
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Entraînement du modèle
    model.fit(X, tags_encoded, epochs=1000)

    # Fonction pour traiter l'entrée de l'utilisateur
    def process_input(user_input):
        # Vous devriez définir cette fonction pour traiter l'entrée de l'utilisateur
        # Par exemple, cela pourrait être la tokenisation et la suppression des stopwords
        return user_input.lower()

    # Fonction pour obtenir la réponse pour une intention donnée
    def get_response_for_intent(intent_tag):
        # Supposons que vous avez un dictionnaire qui mappe les tags d'intention aux réponses
        responses = {
            "salutation": "Bonjour! Comment puis-je vous aider?",
            "formation_info": "Nous offrons une variété de formations. Pour quel domaine êtes-vous intéressé?",
            "profil_entree": "Le profil d'entrée requiert généralement un bac scientifique ou économique.",

            # Ajoutez autant de réponses que nécessaire pour les autres tags
        }

        # Retourner la réponse correspondant au tag d'intention, ou une réponse par défaut
        return responses.get(intent_tag, "Je suis désolé, je ne comprends pas votre demande.")

    # Dialogue avec l'utilisateur
    try:
        while True:
            user_input = input('Vous: ')
            if user_input.lower() == 'quit':
                break

            user_tokens = process_input(user_input)
            user_sequence = pad_sequences(tokenizer.texts_to_sequences([user_tokens]), padding='post', maxlen=max_sequence_length)
            intent_prediction = model.predict(user_sequence)
            intent_index = np.argmax(intent_prediction)
            intent_tag = label_encoder.inverse_transform([intent_index])[0]

            response = get_response_for_intent(intent_tag)
            print("Assistant:", response)

    except Exception as e:
        print(f"Une erreur est survenue: {e}")




def load_intents_from_csv(file_path):
    intents_dict = {}
    with open(file_path, mode='r', encoding='utf-8') as csvfile:
        csv_reader = csv.DictReader(csvfile)
        for row in csv_reader:
            tag = row['intents/tag']
            responses = [response for key, response in row.items() if key.startswith('intents/responses/') and response != ""]
            intents_dict[tag] = responses
    return intents_dict



# Emplacement de votre fichier CSV
csv_file_path = 'orientation_esgis_base.csv'

# Charger les intentions et les réponses du CSV
intents_responses_dict = load_intents_from_csv(csv_file_path)

# Exemple d'utilisation
intent_to_find = "salutation"  # Ce serait l'intent détecté par votre système de NLP
response = get_response_for_intent2(intent_to_find, intents_responses_dict)
print(response)  # C

if __name__ == '__main__':
    call_second_chatbot()
