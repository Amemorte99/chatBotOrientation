from __future__ import annotations
import json
import nltk
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras import optimizers
from keras.optimizers import Adam
from keras.src.layers import Dropout
from sklearn.preprocessing import LabelEncoder  # Import correct LabelEncoder
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from difflib import get_close_matches
from keras.utils import to_categorical


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

stemmer = SnowballStemmer('french')


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


def get_response_for_intent(intent_tag: str, intents: dict) -> str | None:
    for intent in intents["intents"]:
        if intent["tag"] == intent_tag:
            return np.random.choice(intent["responses"])




def chat_bot():
    intents = load_intents('orientation_esgis_base.json')

    # Préparation des motifs et des tags pour l'entraînement
    patterns = []
    tags = []
    for intent in intents["intents"]:
        for pattern in intent["patterns"]:
            patterns.append(pattern)
            tags.append(intent["tag"])

    # Création du tokenizer et conversion des motifs en séquences numériques
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(patterns)
    sequences = tokenizer.texts_to_sequences(patterns)

    # Encodage des tags
    label_encoder = LabelEncoder()
    tags_encoded = label_encoder.fit_transform(tags)

    # Vérification de la correspondance entre les séquences et les tags encodés
    print(f'Nombre de motifs: {len(patterns)}')
    print(f'Nombre de tags: {len(tags)}')
    print(f'Nombre de séquences: {len(sequences)}')
    print(f'Nombre de tags encodés: {len(tags_encoded)}')

    # S'assurer que le nombre de séquences et de tags encodés est le même
    assert len(sequences) == len(tags_encoded), "Le nombre de séquences et de tags encodés ne correspond pas."

    # Padding des séquences pour avoir une longueur uniforme
    max_sequence_length = max([len(seq) for seq in sequences])  # Dynamiquement déterminé à partir des données
    X = pad_sequences(sequences, padding='post', maxlen=max_sequence_length)

    # Définition de l'architecture du modèle
    # model = Sequential()
    # model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=128, input_length=max_sequence_length))
    # model.add(LSTM(100))
    # model.add(Dense(len(set(tags_encoded)), activation='softmax'))

    label_encoder = LabelEncoder()
    tags_encoded = label_encoder.fit_transform(tags)
    tags_encoded = to_categorical(tags_encoded)  # Convert to one-hot encoding

    # ... (No changes until model definition)

    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=(X.shape[1],)))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(tags_encoded.shape[1], activation='softmax'))  # Using the second dimension of tags_encoded

    # Compilation du modèle
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # ... (No changes until the fit function)

    # Conversion des tags encodés en un tableau numpy pour la compatibilité avec Keras est déjà faite
    # Pas besoin de convertir encore une fois car `tags_encoded` est déjà un tableau numpy grâce à `to_categorical`



    # Entraînement du modèle
    model.fit(X, tags_encoded, epochs=100)


    try:
        while True:
            user_input = input('Vous: ')
            if user_input.lower() == 'quit':
                break

            user_tokens = process_input(user_input)
            user_input_sequence = pad_sequences(tokenizer.texts_to_sequences([user_tokens]), padding='post', maxlen=max_sequence_length)

            intent_prediction = model.predict(user_input_sequence)
            intent_index = np.argmax(intent_prediction)
            intent_tag = label_encoder.inverse_transform([intent_index])[0]

            if intent_prediction[0][intent_index] < 0.8:
                closest_match = get_closest_match(user_input, patterns)
                if closest_match:
                    intent_tag = find_best_match(closest_match, patterns)

            response = get_response_for_intent(intent_tag, intents)
            if response:
                print("Assistant:", response)
            else:
                print("Assistant: Je suis désolé, je ne peux pas répondre à cette question pour le moment.")
    except Exception as e:
        print(f"Une erreur est survenue: {e}")

if __name__ == '__main__':
    chat_bot()
