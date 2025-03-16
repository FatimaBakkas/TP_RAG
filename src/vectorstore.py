# src/vectorstore.py
import chromadb
from chromadb.errors import UniqueConstraintError
from chromadb.utils import embedding_functions

class VectorStore:
    def __init__(self, model_name, collection_name="rag_collection"):
        self.client = chromadb.Client()
        try:
            self.collection = self.client.create_collection(
                name=collection_name,
                embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)
            )
        except UniqueConstraintError:
            print(f"La collection {collection_name} existe déjà, chargement de la collection existante.")
            self.collection = self.client.get_collection(name=collection_name)

    def add_documents(self, docs, embeddings):
        for idx, (doc, emb) in enumerate(zip(docs, embeddings)):
            self.collection.add(
                ids=[f"doc_{idx}"],
                embeddings=[emb],
                metadatas=[doc.metadata],
                documents=[doc.page_content]
            )

    def query(self, query_embedding, n_results=5):
        return self.collection.query(query_embeddings=[query_embedding], n_results=n_results)
