import yaml
from langchain_community.llms import HuggingFaceHub  # après l'install de langchain-community
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from src.embedder import Embedder
from src.vectorstore import VectorStore

class QA_System:
    def __init__(self, llm_repo_id, api_key, config_file="config.yaml"):
        # Initialisation du LLM via HuggingFaceHub
        self.llm = HuggingFaceHub(repo_id=llm_repo_id, huggingfacehub_api_token=api_key)
        with open(config_file, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        self.embedder = Embedder(model_name=config["embedding_model"])
        self.vectorstore = VectorStore(model_name=config["embedding_model"],
                                       collection_name=config.get("collection_name", "rag_collection"))
        self.prompt_template = PromptTemplate(
            template="Contexte : {context}\n\nQuestion : {question}\n\nRéponse :",
            input_variables=["context", "question"]
        )
        self.chain = LLMChain(prompt=self.prompt_template, llm=self.llm)

    def get_context(self, query, n_results=5):
        query_embedding = self.embedder.embed_query(query)
        results = self.vectorstore.query(query_embedding, n_results=n_results)
        documents = results.get("documents", [])[:n_results]
        # Transforme chaque document en chaîne s'il s'agit d'une liste
        documents_str = [
            " ".join(doc) if isinstance(doc, list) else doc
            for doc in documents
        ]
        context = "\n\n".join(documents_str)
        return context


    def answer(self, question):
        context = self.get_context(question)
        return self.chain.run(context=context, question=question)
