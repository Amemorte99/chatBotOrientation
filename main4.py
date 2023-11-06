from __future__ import annotations
import json
import numpy as np
import nltk
from difflib import get_close_matches
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Dropout
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

# Téléchargement des ressources de NLTK
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

stemmer = SnowballStemmer('french')

# Fonctions de chargement et d'enregistrement des intentions
def load_intents(file_path: str) -> dict:
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def save_intents(file_path: str, data: dict):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=2, ensure_ascii=False)

# Prétraitement des entrées utilisateur
def process_input(input_text: str) -> list[str]:
    stop_words = set(stopwords.words('french'))
    tokens = [stemmer.stem(token.lower()) for token in word_tokenize(input_text) if token.lower() not in stop_words]
    return tokens

# Trouver la meilleure correspondance
def find_best_match(user_question: str, patterns: list) -> str | None:
    user_tokens = process_input(user_question)
    best_match = None
    best_similarity = 0

    for pattern in patterns:
        pattern_tokens = process_input(pattern)
        similarity = len(set(user_tokens) & set(pattern_tokens)) / len(set(user_tokens) | set(pattern_tokens))

        if similarity > best_similarity:
            best_similarity = similarity
            best_match = pattern

    return best_match if best_similarity >= 0.4 else None

# Obtention de la réponse pour une intention
def get_response_for_intent(intent_tag: str, intents: dict) -> str:
    for intent in intents["intents"]:
        if intent["tag"] == intent_tag:
            return np.random.choice(intent["responses"])
    return "Désolé, je ne suis pas sûr de pouvoir répondre à cela."

# Recherche de la correspondance la plus proche
def get_closest_match(user_input: str, patterns: list) -> str | None:
    close_match = get_close_matches(user_input, patterns, n=1, cutoff=0.8)
    return close_match[0] if close_match else None

# Fonction principale pour l'exécution du chatbot
def chat_bot():
    intents = load_intents('orientation_esgis_base.json')


# Création des paires patterns-tags pour l'entraînement
    pattern_tags_pairs = [(pattern, intent["tag"]) for intent in intents["intents"] for pattern in intent["patterns"]]

# Séparation des patterns et des tags
    patterns = [pair[0] for pair in pattern_tags_pairs]
    tags = [pair[1] for pair in pattern_tags_pairs]

# Tokenisation et création de séquences
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(patterns)
    sequences = tokenizer.texts_to_sequences(patterns)

# Padding des séquences pour l'entrée du modèle
    max_sequence_length = max(len(seq) for seq in sequences)
    X = pad_sequences(sequences, padding='post', maxlen=max_sequence_length)

# Encodage des étiquettes
    label_encoder = LabelEncoder()
    tags_encoded = label_encoder.fit_transform(tags)
    tags_encoded = to_categorical(tags_encoded)

    # Vérification de la cardinalité
    assert len(X) == len(tags_encoded), "Le nombre de séquences d'entrée et d'étiquettes doit être le même."

# ...


    model = Sequential()
    model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=128, input_length=max_sequence_length))
    model.add(LSTM(100))
    model.add(Dropout(0.5))
    model.add(Dense(tags_encoded.shape[1], activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X, tags_encoded, epochs=100)

    while True:
        user_input = input('Vous: ')
        if user_input.lower() == 'quit':
            break

        best_match = find_best_match(user_input, patterns)
        if best_match:
            response = get_response_for_intent(best_match, intents)
            print("Assistant:", response)
        else:
            user_tokens = process_input(user_input)
            user_sequence = pad_sequences(tokenizer.texts_to_sequences([user_tokens]), maxlen=max_sequence_length)
            intent_prediction = model.predict(user_sequence)
            intent_index = np.argmax(intent_prediction)
            intent_tag = label_encoder.inverse_transform([intent_index])[0]
            response = get_response_for_intent(intent_tag, intents)
            print("Assistant:", response)

if __name__ == '__main__':
    chat_bot()
