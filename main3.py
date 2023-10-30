from __future__ import annotations
import json
import nltk
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras import optimizers
from sklearn.preprocessing import LabelEncoder  # Import correct LabelEncoder
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from difflib import get_close_matches

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
    stop_words = set(stopwords.words('english'))
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
            return intent["responses"]


def chat_bot():
    intents = load_intents('orientation_esgis_base.json')
    patterns = [pattern for intent in intents["intents"] for pattern in intent["patterns"]]
    tags = [intent["tag"] for intent in intents["intents"]]

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(patterns)
    sequences = tokenizer.texts_to_sequences(patterns)

    label_encoder = LabelEncoder()
    tags_encoded = label_encoder.fit_transform(tags)

    max_sequence_length = 14
    X = pad_sequences(sequences, padding='post', maxlen=max_sequence_length)

    model = Sequential()
    model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=128, input_length=max_sequence_length))
    model.add(LSTM(100))
    model.add(Dense(len(set(tags_encoded)), activation='softmax'))

    model.compile(optimizer=optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X, tags_encoded, epochs=10)

    while True:
        user_input = input('Vous: ')

        if user_input.lower() == 'quit':
            break

        user_tokens = process_input(user_input)
        user_input_sequence = pad_sequences(tokenizer.texts_to_sequences([user_tokens]), padding='post',
                                            maxlen=max_sequence_length)

        intent_prediction = model.predict(user_input_sequence)

        intent_index = np.argmax(intent_prediction)
        intent_tag = label_encoder.inverse_transform([intent_index])[0]

        if intent_prediction[0][intent_index] < 0.5:
            # Si l'intention prédite a une probabilité inférieure à 0.5, on recherche la correspondance la plus proche
            closest_match = get_closest_match(user_input, patterns)
            if closest_match:
                intent_tag = find_best_match(closest_match, patterns)

        if intent_tag:
            response = get_response_for_intent(intent_tag, intents)
            if response:
                print("Assistant:", np.random.choice(response))
            else:
                print("Assistant: Je suis désolé, je ne peux pas répondre à cette question pour le moment.")
        else:
            print("Assistant: Je suis désolé, je ne comprends pas votre question.")

def chat_bot():
    intents = load_intents('orientation_esgis_base.json')
    patterns = [pattern for intent in intents["intents"] for pattern in intent["patterns"]]
    tags = [intent["tag"] for intent in intents["intents"]]

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(patterns)
    sequences = tokenizer.texts_to_sequences(patterns)

    label_encoder = LabelEncoder()
    tags_encoded = label_encoder.fit_transform(tags)

    max_sequence_length = 14
    X = pad_sequences(sequences, padding='post', maxlen=max_sequence_length)

    model = Sequential()
    model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=128, input_length=max_sequence_length))
    model.add(LSTM(100))
    model.add(Dense(len(set(tags_encoded)), activation='softmax'))

    model.compile(optimizer=optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X, tags_encoded, epochs=10)

    while True:
        user_input = input('Vous: ')

        if user_input.lower() == 'quit':
            break

        user_tokens = process_input(user_input)
        user_input_sequence = pad_sequences(tokenizer.texts_to_sequences([user_tokens]), padding='post',
                                            maxlen=max_sequence_length)

        intent_prediction = model.predict(user_input_sequence)
        intent_index = np.argmax(intent_prediction)
        intent_tag = tags_encoded[intent_index]

        if intent_prediction[0][intent_index] >= 0.5:
            response = get_response_for_intent(intent_tag, intents)
            print('Bot:', np.random.choice(response))
        else:
            named_entities = extract_named_entities(user_input)
            if named_entities:
                for entity in named_entities:
                    closest_match = get_closest_match(entity, patterns)
                    if closest_match:
                        intent_tag = find_best_match(closest_match, patterns)
                        if intent_tag:
                            response = get_response_for_intent(intent_tag, intents)
                            print('Bot:', np.random.choice(response))
                            break
            if not response:
                print("Bot: Je suis désolé, je ne comprends pas. Pouvez-vous reformuler votre question ?")


if __name__ == '__main__':
    chat_bot()
