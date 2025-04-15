import os
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough, RunnableSequence
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_ollama import ChatOllama, OllamaEmbeddings
import shutil
import uvicorn
import logging
from markitdown import MarkItDown
import tempfile
from fastapi import HTTPException
from langchain_community.vectorstores.utils import filter_complex_metadata

import warnings
warnings.simplefilter("ignore", category=Warning)

#logging.basicConfig(level=logging.INFO)

load_dotenv()

# Initialize FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Huggingface embeddings
embeddings = OllamaEmbeddings(model="nomic-embed-text")

CDB_DIRECTORY_PATH = os.getenv("CDB_DIRECTORY_PATH", "app/chromadb")
UPLOADED_FILES_PATH = os.getenv("UPLOADED_FILES_PATH", "app/uploaded_files")

if not os.path.exists(UPLOADED_FILES_PATH):
    os.makedirs(UPLOADED_FILES_PATH)
if not os.path.exists(CDB_DIRECTORY_PATH):
    os.makedirs(CDB_DIRECTORY_PATH)

max_k = int(os.getenv("MAX_K", 5))

def get_store():
    return Chroma(embedding_function=embeddings, persist_directory=CDB_DIRECTORY_PATH)

similarity_prompt = PromptTemplate(
    input_variables=["input", "docs"],
    template="Use the following documents to answer the question, citing the source document and page number, noting that these citations are a requirement if anything from the document is used: {docs}\nQuestion: {input}\nAnswer:"
)

search_prompt = PromptTemplate(
    input_variables=["input"],
    template="Given the query, create a concise semantic search query to obtain relevant documents from a vector database. Return ONLY the relevant query, with no other text or quotation marks: {input}"
)

llm = ChatOllama(model="qwen2.5:7b", temperature=0)

chain = (
    similarity_prompt
    | llm
    | StrOutputParser()
)


def convert_to_markdown(file_path: str) -> str:
    markitdown = MarkItDown()
    try:
        result = markitdown.convert(file_path)
        return result.text_content
    except Exception as e:
        logging.error(f"Markdown conversion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"message": "APP IS RUNNING"}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        file_location = f"{UPLOADED_FILES_PATH}/{file.filename}"
        with open(file_location, "wb") as f:
            shutil.copyfileobj(file.file, f)

        content = convert_to_markdown(file_location)

        with tempfile.NamedTemporaryFile(delete=False, mode='w', encoding='utf-8') as temp_file:
            temp_file.write(content)
            temp_file_path = temp_file.name

        loader = UnstructuredMarkdownLoader(temp_file_path)
        doc_pages = loader.load()
        # Add page numbers to each document (each representing a page)
        for i, page in enumerate(doc_pages):
            page.metadata.update({"page": i + 1})
        os.remove(temp_file_path)

        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        chunks = text_splitter.split_documents(doc_pages)

        # Add filename metadata to each chunk; page metadata is inherited
        for chunk in chunks:
            chunk.metadata.update({"filename": file.filename})

        chunks = filter_complex_metadata(chunks)

        vectorstore = get_store()
        vectorstore.add_documents(chunks)

        return JSONResponse(content={"status": "uploaded"}, status_code=200)

    except Exception as e:
        logging.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search")
async def search(query: str = Form(...)):
    try:
        store = get_store()
        query_refinement = search_prompt.format(input=query)
        refined_query = llm.invoke(query_refinement)
        print(f"Refined query: {refined_query.content}")

        docs = store.similarity_search(refined_query.content, k=max_k)

        docs_json = [{"content": d.page_content, "metadata": d.metadata} for d in docs]

        response = chain.invoke({"input": query, "docs": docs_json})

        return JSONResponse(content={"response": response}, status_code=200)

    except Exception as e:
        logging.error(f"Search error: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/list_documents")
async def list_documents_cdb():
    store = get_store()
    docs = store._collection.get(include=["metadatas", "documents"])
    documents = [{"content": c, "metadata": m} for m, c in zip(docs["metadatas"], docs["documents"])]
    return JSONResponse(content={"documents": documents}, status_code=200)

@app.post("/reset_chromadb")
async def reset_chromadb():
    logging.info("Resetting ChromaDB...")
    try:
        store = get_store()
        store.reset_collection()
        logging.info("ChromaDB reset successfully.")
        return JSONResponse(content={"status": "ChromaDB reset successfully"}, status_code=200)
    except Exception as e:
        logging.error(f"Error resetting ChromaDB: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)