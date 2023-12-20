from flask_cors import CORS
from flask import Flask, request, jsonify
from contextlib import redirect_stdout
import io
import ast  # pour convertir la chaîne de caractères représentant une liste en une liste Python
from chatIA import call_second_chatbot
from principaleChat import call_first_chatbot

app = Flask(__name__)
CORS(app)

@app.route('/chat_bot', methods=['POST'])
def chat():
    data = request.json

    if not data or 'message' not in data:
        return jsonify({'error': 'No message provided'}), 400

    user_input = data['message']
    if user_input.lower() == 'quit':
        return jsonify({'response': 'Chatbot session ended'}), 200

    f = io.StringIO()
    with redirect_stdout(f):
        handled_by_first = call_first_chatbot()
        if not handled_by_first:
            handled_by_second = call_second_chatbot()
            if not handled_by_second:
                return jsonify({'response': "Aucun chatbot n'a pu répondre."})

    response = f.getvalue()

    try:
        # Convertir la chaîne de caractères capturée en liste
        response_list = ast.literal_eval(response.strip())
        # Joindre les éléments de la liste pour former une seule chaîne de caractères
        formatted_response = ' '.join(response_list)
    except:
        # Si la conversion échoue, renvoyer la réponse telle quelle
        formatted_response = response.strip()

    return jsonify({'response': formatted_response})

if __name__ == '__main__':
    app.run(debug=True)
