import os
import fitz
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Tuple

class RAGPipeline:
    def __init__(self, data_folder="data/sample_docs"):
        self.data_folder = data_folder
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.documents: List[str] = []
        self.doc_sources: List[Tuple[str, int]] = []  # (filename, chunk_id)
        self.index = None
        self._load_documents()
        self._build_index()

    def _load_documents(self):
        self.documents.clear()
        self.doc_sources.clear()

        for filename in os.listdir(self.data_folder):
            if filename.startswith("."):
                continue

            filepath = os.path.join(self.data_folder, filename)
            if filename.endswith(".txt"):
                text = open(filepath, "r", encoding="utf-8").read()
            elif filename.endswith(".pdf"):
                text = self._read_pdf(filepath)
            else:
                continue

            chunks = self._split_into_chunks(text)
            for i, chunk in enumerate(chunks):
                self.documents.append(chunk)
                self.doc_sources.append((filename, i))

    def _read_pdf(self, path):
        try:
            text = ""
            with fitz.open(path) as doc:
                for page in doc:
                    text += page.get_text()
            return text
        except Exception as e:
            print(f"Error reading {path}: {e}")
            return ""

    def _split_into_chunks(self, text, max_chars=500):
        # Simple paragraph-based chunking
        paragraphs = text.split("\n\n")
        chunks = []
        for p in paragraphs:
            p = p.strip()
            if p:
                while len(p) > max_chars:
                    chunks.append(p[:max_chars])
                    p = p[max_chars:]
                if p:
                    chunks.append(p)
        return chunks

    def _build_index(self):
        if not self.documents:
            raise ValueError("No documents to index.")

        embeddings = self.embedding_model.encode(self.documents).astype("float32")
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)

    def _retrieve(self, query, top_k=2):
        query_emb = self.embedding_model.encode([query]).astype("float32")
        _, indices = self.index.search(query_emb, top_k)
        return [(self.documents[i], self.doc_sources[i]) for i in indices[0]]

    def query(self, question):
        retrieved = self._retrieve(question)
        context = "\n".join([doc for doc, _ in retrieved])
        prompt = f"""Answer the question based on the context below:

Context:
{context}

Question: {question}
Answer:"""
        answer = self._generate(prompt)
        return {
            "answer": answer,
            "sources": [{"filename": f, "chunk": c} for _, (f, c) in retrieved]
        }

    def _generate(self, prompt):
        import requests
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "gemma3:1b", "prompt": prompt, "stream": False}
        )
        data = response.json()
        return data.get("response", "").strip()

    def add_file_and_reindex(self, filename: str, content: bytes):
        path = os.path.join(self.data_folder, filename)
        with open(path, "wb") as f:
            f.write(content)
        self._load_documents()
        self._build_index()
