# TP_RAG - Retrieval Augmented Generation

Dans le cadre du TP RAG, nous avons implémenté un **système de recherche augmentée par génération (RAG)**.  
Ce système RAG permet d'extraire des informations à partir de documents et de générer des réponses basées sur un llm.

---

## Structure du projet

TP_RAG/
├── cli.py                # Interface en ligne de commande
├── config.yaml           # Fichier de configuration (modèles, API keys, chemins)
├── requirements.txt      # Liste des dépendances du projet
├── data/                 # Dossier contenant les fichiers PDF à indexer
└── src/                  # Code source du projet
    ├── __init__.py
    ├── loader.py         # Chargement des fichiers PDF
    ├── markdown_splitter.py  # Découpage en chunks optimisés
    ├── indexer.py        # Indexation des documents
    ├── embedder.py       # Génération des embeddings
    ├── vectorstore.py    # Stockage et récupération des embeddings
    ├── qa_system.py      # Système de question-réponse
    ├── evaluator.py      # Évaluation des réponses générées
    └── chatbot.py        # Chatbot interactif

---

## Installation des dépendances et lancement du chat :

1. **Cloner le repository** :
   ```bash
   git clone <URL-du-repo>
   cd TP_RAG
2. **Installer les dépendances** :
   ```bash
   pip install -r requirements.txt
3. **Indexation des documents dans le folder data/** :
   ```bash
   python cli.py index
4. **Lancement du chat et qa par rapport aux documents indexés** :
   ```bash
   python cli.py chat
   

