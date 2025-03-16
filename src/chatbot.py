import yaml
from langchain_community.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from src.embedder import Embedder
from src.vectorstore import VectorStore

class ChatBot:
    def __init__(self, llm_repo_id, api_key, config_file="config.yaml"):
        # Initialisation du LLM via HuggingFaceHub
        self.llm = HuggingFaceHub(repo_id=llm_repo_id, huggingfacehub_api_token=api_key)
        # Charger la configuration
        with open(config_file, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        # Initialiser l'embedder et le vector store
        self.embedder = Embedder(model_name=config["embedding_model"])
        self.vectorstore = VectorStore(model_name=config["embedding_model"],
                                       collection_name=config.get("collection_name", "rag_collection"))
        # Historique de la conversation (liste de chaînes)
        self.conversation_history = []
        # Template de prompt incluant l'historique, le contexte et la question
        self.prompt_template = PromptTemplate(
            template="""Tu es un assistant spécialisé en NLP. Réponds à la question suivante en utilisant uniquement le contexte fourni et en tenant compte de l'historique des échanges. Si l'information ne figure pas dans le contexte, indique "Je ne sais pas".

Historique de la conversation :
{history}

Contexte pertinent :
{context}

Question actuelle :
{question}

Réponse détaillée :""",
            input_variables=["history", "context", "question"]
        )
        # Chaîne LLM qui utilisera le prompt
        self.chain = LLMChain(prompt=self.prompt_template, llm=self.llm)

    def get_context(self, query, n_results=5):
        """Récupère les documents les plus pertinents depuis le vector store."""
        query_embedding = self.embedder.embed_query(query)
        results = self.vectorstore.query(query_embedding, n_results=n_results)
        documents = results.get("documents", [])[:n_results]
        # Transformation : aplatir les listes en chaînes de caractères
        documents_str = [
            " ".join(doc) if isinstance(doc, list) else doc
            for doc in documents
        ]
        context = "\n\n".join(documents_str)
        return context if context.strip() else "Aucun contexte trouvé."

    def chat(self, question):
        """Génère une réponse en tenant compte de l'historique et met à jour l'historique."""
        context = self.get_context(question)
        # On ne prend que les 5 derniers échanges pour limiter la taille du prompt
        history_text = "\n".join(self.conversation_history[-5:])
        # Appeler la chaîne pour générer une réponse
        response = self.chain.invoke({"history": history_text, "context": context, "question": question})
        response_text = response.get("text", "").strip()
        # Mettre à jour l'historique
        self.conversation_history.append("Utilisateur : " + question)
        self.conversation_history.append("Assistant : " + response_text)
        return response_text