from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from app.rag_pipeline import RAGPipeline

app = FastAPI()
rag = RAGPipeline()

class Query(BaseModel):
    question: str

@app.post("/ask")
def ask_question(query: Query):
    result = rag.query(query.question)
    return {
        "answer": result["answer"],
        "sources": result["sources"]
    }

@app.post("/upload")
def upload_file(file: UploadFile = File(...)):
    if not (file.filename.endswith(".txt") or file.filename.endswith(".pdf")):
        return {"error": "Only .txt and .pdf files are supported."}
    
    content = file.file.read()
    rag.add_file_and_reindex(file.filename, content)
    return {"message": f"File '{file.filename}' uploaded and indexed successfully."}

@app.get("/")
def root():
    return {"message": "Ollama-powered RAG is running!"}