# src/indexer.py
from src.loader import DocumentLoader
from src.markdown_splitter import MarkdownSplitter
from src.embedder import Embedder
from src.vectorstore import VectorStore
from src.markdown_splitter import MarkdownSplitter


class Indexer:
    def __init__(self, config):
        self.file_paths = config["file_paths"]
        self.embedding_model = config["embedding_model"]
        self.max_chunk_size = config.get("chunk_size", 1000)
        self.collection_name = config.get("collection_name", "rag_collection")
        self.embedder = Embedder(model_name=self.embedding_model)
        # Utilisation du MarkdownSplitter pour préserver la structure
        self.splitter = MarkdownSplitter(max_chunk_size=self.max_chunk_size)
        self.vectorstore = VectorStore(model_name=self.embedding_model, collection_name=self.collection_name)

    def run(self):
        all_chunks = []
        for file_path in self.file_paths:
            loader = DocumentLoader(file_path)
            docs = loader.load()
            # Utiliser le MarkdownSplitter pour découper le document tout en préservant la structure Markdown
            chunks = self.splitter.split_documents(docs)
            all_chunks.extend(chunks)
        embeddings = self.embedder.embed_documents(all_chunks)
        self.vectorstore.add_documents(all_chunks, embeddings)
