from __future__ import annotations
import json
import nltk
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from difflib import get_close_matches

from tensorflow.python.keras.layers import LSTM
from tensorflow.python.keras.legacy_tf_layers.core import Dense
from tensorflow.python.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

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
    stop_words = set(stopwords.words('english'))
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

    user_tokens = [stemmer.stem(token.lower()) for token in word_tokenize(user_question) if token.lower() not in stop_words]
    best_match = None
    best_similarity = 0

    for pattern in patterns:
        pattern_tokens = [stemmer.stem(token.lower()) for token in word_tokenize(pattern) if token.lower() not in stop_words]
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
    print(nltk.__version__)
    intents: dict = load_intents('orientation_esgis_base.json')
    training_data = json.load(open('orientation_esgis_base.json'))  # Chargement des données d'entrainement

    inputs = []
    targets = []

    for data in training_data['intents']:
        tag = data['tag']
        patterns = data['patterns']
        for pattern in patterns:
            inputs.append(pattern)
            targets.append(tag)

    # Préparation des données d'entrée et sortie
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(inputs)
    sequences = tokenizer.texts_to_sequences(inputs)
    X = pad_sequences(sequences)
    y = np.array(targets)

    # Définition du modèle LSTM
    model = Sequential()

    model.add(LSTM(100, input_shape=(X.shape[1], X.shape[2])))
    model.add(Dense(1, activation='sigmoid'))

    # Compilation et entraînement du modèle
    model.compile(...)
    model.fit(X, y)

    while True:
        user_input: str = input('Vous: ')

        if user_input.lower() == 'quit':
            break

        best_match: str | None = find_best_match(user_input, inputs)

        if not best_match:
            best_match = get_closest_match(user_input, inputs)

        if best_match:
            matching_intents = [intent for intent in intents["intents"] if best_match in intent["patterns"]]
            if matching_intents:
                responses = []
                for intent in matching_intents:
                    response = get_response_for_intent(intent["tag"], intents)
                    if response:
                        responses.extend(response)
                        if responses:
                            print("Bot:")
                            for response in responses:
                                print(f"- {response}")
                        else:
                            named_entities = extract_named_entities(user_input)
                            if named_entities:
                                print(
                                    f'Bot: Je vois que vous parlez de {", ".join(named_entities)}. Pouvez-vous fournir plus de détails ?')
                            else:
                                print('Bot: Je ne connais pas la réponse. Pouvez-vous m\'apprendre ?')
                            new_answer: str = input('Tapez la réponse ou "Skip" pour passer : ')

                            if new_answer.lower() != 'skip':
                                # Ajouter la nouvelle intention au JSON
                                intent_tag = "custom_intent"
                                intents["intents"].append(
                                    {"tag": intent_tag, "patterns": [user_input], "responses": [new_answer]})
                                save_intents('intents.json', intents)
                                print('Bot: Merci ! J\'ai appris une nouvelle réponse')



if __name__ == '__main__':
    chat_bot()