
from __future__ import annotations

import nltk
import json
from difflib import get_close_matches
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk


nltk.download('punkt')  # Télécharge les modèles tokenizer
nltk.download('stopwords')  # Télécharge les mots vides (stopwords) en français
nltk.download('averaged_perceptron_tagger')  # Télécharge le modèle POS tagger
nltk.download('maxent_ne_chunker')  # Télécharge le modèle de Named Entity Recognition (NER)
nltk.download('words')  # Télécharge les mots communs en anglais (nécessaire pour le POS tagger)

stemmer = SnowballStemmer('french')
from flask import Flask, request, jsonify
import nltk

app = Flask(__name__)

def load_intents(file_path: str) -> dict:
    with open(file_path, 'r', encoding='utf-8') as file:
        data: dict = json.load(file)
    return data


def save_intents(file_path: str, data: dict):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


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


def get_response_for_intent1(intent_tag: str, intents: dict) -> str | None:
    for intent in intents["intents"]:
        if intent["tag"] == intent_tag:
            return intent["responses"]


def get_closest_match(user_input: str, patterns: list) -> str | None:
    close_match = get_close_matches(user_input, patterns, n=1, cutoff=0.8)
    if close_match:
        return close_match[0]
    else:
        return None


def extract_named_entities(text: str) -> list:
    tagged_tokens = pos_tag(word_tokenize(text))
    named_entities = ne_chunk(tagged_tokens)
    entities = []
    for subtree in named_entities.subtrees():
        if subtree.label() == 'NE':
            entity = ' '.join(word for word, tag in subtree.leaves())
            entities.append(entity)
    return entities




# Chargez vos intentions ici, si nécessaire
intents = load_intents('orientation_esgis_base.json')

@app.route('/chat', methods=['POST'])
def call_first_chatbot():
    user_input = request.json.get('message')

    if not user_input:
        return jsonify({'error': 'No message provided'}), 400

    if user_input.lower() == 'quit':
        return jsonify({'response': 'Chatbot session ended'}), 200

    best_match = find_best_match(user_input, [p for intent in intents["intents"] for p in intent["patterns"]])

    if not best_match:
        best_match = get_closest_match(user_input, [p for intent in intents["intents"] for p in intent["patterns"]])

    responses = []
    if best_match:
        matching_intents = [intent for intent in intents["intents"] if best_match in intent["patterns"]]
        for intent in matching_intents:
            response = get_response_for_intent1(intent["tag"], intents)
            if response:
                print(f" {response}")
                responses.extend(response)
    else:
        named_entities = extract_named_entities(user_input)
        if named_entities:
            return jsonify({
                               'response': f'Je vois que vous parlez de {", ".join(named_entities)}. Pouvez-vous fournir plus de détails ?'})
        else:
            return jsonify({'response': 'Je ne connais pas la réponse. Pouvez-vous m\'apprendre ?'})

    return jsonify({'responses': responses})


if __name__ == '__main__':
    app.run(debug=True)
