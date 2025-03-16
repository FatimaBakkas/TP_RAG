#!/usr/bin/env python
import argparse
import yaml
from src.indexer import Indexer
from src.qa_system import QA_System
from src.evaluator import evaluate_answer
from src.chatbot import ChatBot

def load_config(config_file="config.yaml"):
    with open(config_file, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    config = load_config()

    parser = argparse.ArgumentParser(description="TP Retrieval Augmented Generation (RAG)")
    subparsers = parser.add_subparsers(dest="command", help="Commandes disponibles")

    subparsers.add_parser("index", help="Indexe les documents PDF")
    
    qa_parser = subparsers.add_parser("qa", help="Question/Réponse")
    qa_parser.add_argument("question", type=str, help="La question à poser")

    eval_parser = subparsers.add_parser("eval", help="Évaluer une réponse générée")
    eval_parser.add_argument("question", type=str, help="La question posée")
    eval_parser.add_argument("reference", type=str, help="La réponse de référence")

    subparsers.add_parser("chat", help="Lancer le chatbot en mode interactif")

    args = parser.parse_args()

    if args.command == "index":
        indexer = Indexer(config)
        indexer.run()
        print("Indexation terminée.")

    elif args.command == "qa":
        qa = QA_System(config["llm_repo_id"], config["api_key"])
        answer = qa.answer(args.question)
        print("Réponse :\n", answer)

    elif args.command == "eval":
        qa = QA_System(config["llm_repo_id"], config["api_key"])
        generated = qa.answer(args.question)
        score = evaluate_answer(generated, args.reference)
        print("Réponse générée :\n", generated)
        print("Score de similarité :", score)

    elif args.command == "chat":
        chatbot = ChatBot(config["llm_repo_id"], config["api_key"])
        print("Bienvenue dans le chatbot. Tapez 'exit' pour quitter.")
        while True:
            user_input = input("Vous : ")
            if user_input.lower() == "exit":
                break
            response = chatbot.chat(user_input)
            print("Chatbot :", response)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()