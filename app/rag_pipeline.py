import os
import fitz # PyMuPDF
import numpy as np
import faiss
import pickle
import json
from sentence_transformers import SentenceTransformer
from typing import List, Tuple
from langchain_text_splitters import RecursiveCharacterTextSplitter
import requests

class RAGPipeline:
    def __init__(self, data_folder="data/sample_docs", index_folder="data/faiss_index"):
        self.data_folder = data_folder
        self.index_folder = index_folder
        os.makedirs(self.data_folder, exist_ok=True)
        os.makedirs(self.index_folder, exist_ok=True)

        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.documents: List[str] = []
        self.doc_sources: List[Tuple[str, int]] = []
        self.index = None

        if not self._load_index():
            if os.listdir(self.data_folder):
                self._load_documents()
                self._build_index()
                self._save_index()
            else:
                print("No documents found in data folder. Index will be created upon first upload.")

    def _load_documents(self):
        self.documents.clear()
        self.doc_sources.clear()

        for filename in os.listdir(self.data_folder):
            if filename.startswith("."):
                continue

            filepath = os.path.join(self.data_folder, filename)
            text = ""

            if filename.endswith(".txt"):
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        text = f.read()
                    print(f"[TXT] Loaded {filename}, {len(text)} characters")
                except Exception as e:
                    print(f"Error reading TXT {filename}: {e}")

            elif filename.endswith(".pdf"):
                text = self._read_pdf(filepath)
                print(f"[PDF] Loaded {filename}, {len(text)} characters")

            if not text.strip():
                continue

            chunks = self._split_into_chunks(text)
            for i, chunk in enumerate(chunks):
                self.documents.append(chunk)
                self.doc_sources.append((filename, i))

        print(f"✅ Total chunks indexed: {len(self.documents)}")

    def _read_pdf(self, path):
        try:
            text = ""
            with fitz.open(path) as doc:
                for page in doc:
                    text += page.get_text()
            return text.strip()
        except Exception as e:
            print(f"Error reading PDF {path}: {e}")
            return ""

    def _split_into_chunks(self, text, chunk_size=500, chunk_overlap=50):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        return text_splitter.split_text(text)

    def _build_index(self):
        if not self.documents:
            raise ValueError("No documents to index.")

        embeddings = self.embedding_model.encode(self.documents).astype("float32")
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)
        print("📦 FAISS index built successfully")

    def _save_index(self):
        if self.index:
            faiss.write_index(self.index, os.path.join(self.index_folder, "faiss.index"))
            with open(os.path.join(self.index_folder, "documents.pkl"), "wb") as f:
                pickle.dump(self.documents, f)
            with open(os.path.join(self.index_folder, "sources.pkl"), "wb") as f:
                pickle.dump(self.doc_sources, f)
            print("💾 FAISS index and documents saved.")

    def _load_index(self):
        try:
            index_path = os.path.join(self.index_folder, "faiss.index")
            docs_path = os.path.join(self.index_folder, "documents.pkl")
            sources_path = os.path.join(self.index_folder, "sources.pkl")
            
            if not all(os.path.exists(p) for p in [index_path, docs_path, sources_path]):
                print("❌ One or more index files are missing.")
                return False

            self.index = faiss.read_index(index_path)
            with open(docs_path, "rb") as f:
                self.documents = pickle.load(f)
            with open(sources_path, "rb") as f:
                self.doc_sources = pickle.load(f)
            print("✅ FAISS index and documents loaded from disk.")
            return True
        except Exception as e:
            print(f"❌ Error loading index: {e}")
            return False
        
    def _process_and_add_document(self, filename: str):
        filepath = os.path.join(self.data_folder, filename)
        text = ""
        
        if filename.endswith(".txt"):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    text = f.read()
                print(f"[TXT] Processing new file {filename}, {len(text)} characters")
            except Exception as e:
                print(f"Error reading TXT {filename}: {e}")
        
        elif filename.endswith(".pdf"):
            text = self._read_pdf(filepath)
            print(f"[PDF] Processing new file {filename}, {len(text)} characters")

        if not text.strip():
            print(f"❌ Skipped empty file: {filename}")
            return

        chunks = self._split_into_chunks(text)
        new_embeddings = self.embedding_model.encode(chunks).astype("float32")
        self.index.add(new_embeddings)
        
        for i, chunk in enumerate(chunks):
            self.documents.append(chunk)
            self.doc_sources.append((filename, i))
            
        print(f"➕ Added {len(chunks)} new chunks from {filename}.")

    def _retrieve(self, query, top_k=2):
        if self.index is None:
            raise ValueError("Index is not loaded or built.")
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
        # This is the non-streaming version
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": "gemma3:1b", "prompt": prompt, "stream": False}
            )
            response.raise_for_status()
            data = response.json()
            return data.get("response", "").strip()
        except requests.exceptions.RequestException as e:
            print(f"❌ Error during Ollama API call: {e}")
            return f"Error: Failed to connect to Ollama server. Please ensure the model is running. {e}"

    def add_file_and_reindex(self, filename: str, content: bytes):
        path = os.path.join(self.data_folder, filename)
        
        try:
            with open(path, "wb") as f:
                f.write(content)
            print(f"📁 Uploaded and saved: {filename}")
        except Exception as e:
            print(f"❌ Failed to write file {filename}: {e}")
            return {"status": "error", "message": str(e)}

        if self.index is None:
            print("No existing index. Building from scratch...")
            self._load_documents()
            self._build_index()
        else:
            try:
                self._process_and_add_document(filename)
            except Exception as e:
                print(f"❌ Failed to process and index new file {filename}: {e}")
                os.remove(path)
                return {"status": "error", "message": f"Failed to process file: {e}"}

        self._save_index()
        return {"status": "success", "message": f"{filename} uploaded and indexed."}