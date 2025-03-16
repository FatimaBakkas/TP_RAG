import streamlit as st
import yaml
from src.indexer import Indexer
from src.qa_system import QA_System
from src.evaluator import evaluate_answer
from src.chatbot import ChatBot

def load_config(config_file="config.yaml"):
    with open(config_file, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# Charger la configuration
config = load_config()
qa_system = QA_System(config["llm_repo_id"], config["api_key"])
chatbot = ChatBot(config["llm_repo_id"], config["api_key"])

# Interface Streamlit
st.title("Chatbot RAG")
st.write("Bienvenue dans l'interface de chatbot utilisant Retrieval Augmented Generation (RAG).")

# Stocker les messages du chat
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Afficher l'historique des messages
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Entrée utilisateur
user_input = st.chat_input("Posez une question...")
if user_input:
    # Ajouter la question de l'utilisateur à l'historique
    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)
    
    # Obtenir la réponse du chatbot
    response = chatbot.chat(user_input)
    st.session_state["messages"].append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.write(response)
