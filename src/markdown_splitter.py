# src/markdown_splitter.py
import re
from langchain.docstore.document import Document

class MarkdownSplitter:
    def __init__(self, max_chunk_size=1000):
        """
        max_chunk_size définit le nombre maximum de caractères par chunk.
        """
        self.max_chunk_size = max_chunk_size

    def split_text(self, text):
        """
        Découpe le texte Markdown en blocs en utilisant les titres Markdown comme séparateurs.
        Pour les blocs trop volumineux, un second découpage est appliqué.
        """
        # Séparation par titres Markdown (ex: "#", "##", etc.)
        blocks = re.split(r'(?=\n#+\s)', text)
        chunks = []
        for block in blocks:
            block = block.strip()
            if not block:
                continue
            if len(block) > self.max_chunk_size:
                chunks.extend(self._split_large_block(block))
            else:
                chunks.append(block)
        return chunks

    def _split_large_block(self, block):
        """
        Découpe un bloc volumineux en utilisant les doubles sauts de ligne pour respecter la structure.
        """
        parts = block.split('\n\n')
        chunks = []
        current_chunk = ""
        for part in parts:
            # On ajoute le séparateur pour conserver la lisibilité du Markdown
            if len(current_chunk) + len(part) + 2 <= self.max_chunk_size:
                current_chunk = f"{current_chunk}\n\n{part}" if current_chunk else part
            else:
                chunks.append(current_chunk)
                current_chunk = part
        if current_chunk:
            chunks.append(current_chunk)
        return chunks

    def split_documents(self, documents):
        """
        Applique le découpage sur une liste d'objets Document (issu de LangChain) 
        en préservant leurs métadonnées.
        """
        new_docs = []
        for doc in documents:
            chunks = self.split_text(doc.page_content)
            for chunk in chunks:
                new_docs.append(Document(page_content=chunk, metadata=doc.metadata))
        return new_docs
