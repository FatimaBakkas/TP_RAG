from langchain_huggingface import HuggingFaceEmbeddings

class Embedder:
    def __init__(self, model_name):
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)

    def embed_documents(self, docs):
        return [self.embeddings.embed_query(doc.page_content) for doc in docs]

    def embed_query(self, query):
        return self.embeddings.embed_query(query)
