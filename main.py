from __future__ import annotations

import nltk
import json
from difflib import get_close_matches
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')  # Télécharge les modèles tokenizer
nltk.download('averaged_perceptron_tagger')  # Télécharge les modèles POS tagger
nltk.download('stopwords')
nltk.download('wordnet')


def load_knowledge_base(file_path: str) -> dict:
    with open(file_path, 'r') as file:
        data: dict = json.load(file)
    return data


def save_knowledge_base(file_path: str, data: dict):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=2)


def find_best_match(user_question: str, questions: list) -> str | None:
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('french'))

    user_tokens = [lemmatizer.lemmatize(token.lower()) for token in word_tokenize(user_question) if
                   token.lower() not in stop_words]
    best_match = None
    best_similarity = 0

    for question in questions:
        question_tokens = [lemmatizer.lemmatize(token.lower()) for token in word_tokenize(question) if
                           token.lower() not in stop_words]
        similarity = len(set(user_tokens) & set(question_tokens)) / len(set(user_tokens) | set(question_tokens))

        if similarity > best_similarity:
            best_similarity = similarity
            best_match = question

    if best_similarity >= 0.4:
        return best_match
    else:
        return None


def get_answer_for_question(question: str, knowledge_base: dict) -> str | None:
    for q in knowledge_base["questions"]:
        if q["question"] == question:
            return q["answers"][q["correctAnswer"]]


def get_closest_match(user_input: str, questions: list) -> str | None:
    close_match = get_close_matches(user_input, questions, n=1, cutoff=0.8)
    if close_match:
        return close_match[0]
    else:
        return None


def chat_bot():
    print(nltk.__version__)
    knowledge_base: dict = load_knowledge_base('orientation_esgis_base.json')

    while True:
        user_input: str = input('Vous: ')

        if user_input.lower() == 'quit':
            break

        best_match: str | None = find_best_match(user_input, [q["question"] for q in knowledge_base["questions"]])

        if not best_match:
            best_match = get_closest_match(user_input, [q["question"] for q in knowledge_base["questions"]])

        if best_match:
            answer: str = get_answer_for_question(best_match, knowledge_base)
            print(f'Bot: {answer}')
        else:
            print('Bot: Je ne connais pas la réponse. Pouvez-vous m \'apprendre ?')
            new_answer: str = input('Tapez la réponse ou "Skip" pour passer : ')

            if new_answer.lower() != 'skip':
                knowledge_base["questions"].append({"question": user_input, "answers": new_answer})
                save_knowledge_base('orientation_esgis_base.json', knowledge_base)
                print('Bot: Merci ! J\'ai appris une nouvelle réponse !')


if __name__ == '__main__':
    chat_bot()