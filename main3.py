from __future__ import annotations
import json
import nltk
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Dropout
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from difflib import get_close_matches

# Téléchargement des ressources nécessaires de NLTK
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

# Initialisation du stemmer
stemmer = SnowballStemmer('french')

# Fonctions pour charger et enregistrer les intentions
def load_intents(file_path: str) -> dict:
    with open(file_path, 'r', encoding='utf-8') as file:
        data: dict = json.load(file)
    return data

def save_intents(file_path: str, data: dict):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=2, ensure_ascii=False)

# Traitement de l'entrée utilisateur
def process_input(input_text: str) -> list[str]:
    stop_words = set(stopwords.words('french'))
    tokens = [stemmer.stem(token.lower()) for token in word_tokenize(input_text) if token.lower() not in stop_words]
    return tokens

# Extraction d'entités nommées
def extract_named_entities(text: str) -> list[str]:
    tagged_tokens = pos_tag(word_tokenize(text), lang='french')
    named_entities = ne_chunk(tagged_tokens)
    entities = []
    for subtree in named_entities.subtrees():
        if subtree.label() == 'NE':
            entity = ' '.join(word for word, tag in subtree.leaves())
            entities.append(entity)
    return entities

# Recherche de correspondance proche
def get_closest_match(user_input: str, patterns: list[str]) -> str | None:
    close_match = get_close_matches(user_input, patterns, n=1, cutoff=0.8)
    return close_match[0] if close_match else None

# Trouver la meilleure correspondance
def find_best_match(user_question: str, patterns: list[str]) -> str | None:
    stop_words = set(stopwords.words('french'))
    user_tokens = [stemmer.stem(token.lower()) for token in word_tokenize(user_question) if token.lower() not in stop_words]
    best_match = None
    best_similarity = 0
    for pattern in patterns:
        pattern_tokens = [stemmer.stem(token.lower()) for token in word_tokenize(pattern) if token.lower() not in stop_words]
        similarity = len(set(user_tokens) & set(pattern_tokens)) / (len(set(user_tokens) | set(pattern_tokens)) + 1e-9)
        if similarity > best_similarity:
            best_similarity = similarity
            best_match = pattern
    return best_match if best_similarity >= 0.4 else None

# Récupération de la réponse pour une intention
def get_response_for_intent(intent_tag: str, intents: dict) -> str | None:
    for intent in intents["intents"]:
        if intent["tag"] == intent_tag:
            return np.random.choice(intent["responses"])

# Définition et entraînement du modèle de chatbot
def chat_bot():
    intents = load_intents('orientation_esgis_base.json')
    patterns, tags = [], []
    for intent in intents["intents"]:
        for pattern in intent["patterns"]:
            patterns.append(pattern)
            tags.append(intent["tag"])

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(patterns)
    sequences = tokenizer.texts_to_sequences(patterns)

    label_encoder = LabelEncoder()
    tags_encoded = label_encoder.fit_transform(tags)
    tags_encoded = to_categorical(tags_encoded)

    max_sequence_length = max(len(seq) for seq in sequences)
    X = pad_sequences(sequences, padding='post', maxlen=max_sequence_length)

    model = Sequential()
    model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=128, input_length=max_sequence_length))
    model.add(LSTM(100))
    model.add(Dropout(0.5))
    model.add(Dense(tags_encoded.shape[1], activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X, tags_encoded, epochs=100)

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
            response = get_response_for_intent(intent_tag, intents)
            print("Assistant:", response if response else "Je suis désolé, je ne peux pas répondre à cette question pour le moment.")
    except Exception as e:
        print(f"Une erreur est survenue: {e}")

if __name__ == '__main__':
    chat_bot()
