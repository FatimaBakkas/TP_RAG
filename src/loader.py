# src/loader.py
from langchain_community.document_loaders import PyMuPDFLoader


class DocumentLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def load(self):
        loader = PyMuPDFLoader(self.file_path)
        return loader.load()
