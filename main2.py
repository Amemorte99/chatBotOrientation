# Fonction pour le premier chatbot
from main import get_response_for_intent1, load_intents
from main3 import get_response_for_intent2


def call_first_chatbot(user_input, intents):
    # Logique pour trouver la meilleure correspondance avec le premier chatbot
    responses = get_response_for_intent1(user_input, intents)
    if responses:
        print("Chatbot 1:")
        for response in responses:
            print(f"- {response}")
        return True  # Retourner True signifie que le chatbot a trouvé une réponse
    return False  # Retourner False signifie qu'il faut appeler le second chatbot

# Fonction pour le second chatbot
def call_second_chatbot(user_input, intents):
    # Logique pour trouver la meilleure correspondance avec le second chatbot
    responses = get_response_for_intent2(user_input, intents)
    if responses:
        print("Chatbot 2:")
        for response in responses:
            print(f"- {response}")

# La fonction principale qui gère la session de chat
def chat_session():
    intents = load_intents('orientation_esgis_base.json')

    while True:
        user_input = input('Vous: ')
        if user_input.lower() == 'quit':
            break

        if not call_first_chatbot(user_input, intents):
            call_second_chatbot(user_input, intents)

if __name__ == '__main__':
    chat_session()


