import os
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import CharacterTextSplitter
import shutil
import uvicorn
import logging
from markitdown import MarkItDown
import tempfile
from fastapi import HTTPException
import nltk
from langchain_community.vectorstores.utils import filter_complex_metadata
import hashlib
from copy import deepcopy

logging.basicConfig(level=logging.INFO)

load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware to allow requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# TODO R.M./M.T.: implement huggingface embeddings
embeddings = HuggingFaceEmbeddings()

CDB_DIRECTORY_PATH = os.getenv("CDB_DIRECTORY_PATH", "app/chromadb")
UPLOADED_FILES_PATH = os.getenv("UPLOADED_FILES_PATH", "app/uploaded_files")

max_k = int(os.getenv("MAX_K", 10000))

def get_store():
    return Chroma(embedding_function=embeddings, persist_directory=CDB_DIRECTORY_PATH)

similarity_prompt = PromptTemplate(
    input_variables=["input", "docs"],
    template="Use the following documents to answer the question, remembering to cite the source of the answer as it relates to the documents: {docs}\nQuestion: {input}\nAnswer:"
)

# TODO R.M.: implement huggingface LegalBERT
llm = ...

def convert_to_markdown(file_path: str) -> str:
    logging.info(f"Converting file to markdown: {file_path}")
    markitdown = MarkItDown()
    """Convert document to markdown format"""
    try:
        result = markitdown.convert(file_path)
        logging.info("Conversion to markdown successful")
        return result.text_content
    except Exception as e:
        logging.error(f"Error converting document to markdown: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    logging.info("Endpoint '/' called")
    return {"message": "APP IS RUNNING"}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    logging.info("Endpoint '/upload' called")
    try:
        logging.info(f"Uploading file: {file.filename}")
        file_location = f"{UPLOADED_FILES_PATH}/{file.filename}"
        with open(file_location, "wb") as f:
            shutil.copyfileobj(file.file, f)
        logging.info(f"Saving file to {file_location}")
        content = convert_to_markdown(file_location)
        logging.info(f"Content: {content}")

        # Create temporary file to ensure nothing is lost due to error
        with tempfile.NamedTemporaryFile(delete=False, mode='w', encoding='utf-8') as temp_file:
            temp_file.write(content)
            temp_file_path = temp_file.name
            logging.info(f"Temporary file path: {temp_file_path}")
        loader = UnstructuredMarkdownLoader(temp_file_path)
        doc = loader.load()
        os.remove(temp_file_path)

        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        chunks = text_splitter.split(doc)

        # TODO (anyone): Save chunks with page number and filename in metadata
        for chunk in chunks:
            chunk.metadata.update({
                "filename": file.filename,
            })
        chunks = filter_complex_metadata(chunks)
        vectorstore = get_store()
        try:
            vectorstore.add_documents(chunks)
        except Exception as e:
            logging.error(f"Error adding documents to vectorstore: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logging.error(f"Error uploading file: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search")
async def search(query: str = Form(...)):
    logging.info(f"Search initiated with query: '{query}'")
    try:
        store = get_store()
        # Perform similarity search without filtering by group or collection
        docs = store.similarity_search(query, k=max_k)
        logging.info(f"Retrieved {len(docs)} chunks from similarity search.")

        docs_json = [{"content": d.page_content, "metadata": d.metadata} for d in docs]
        response = llm.invoke({"input": query, "docs": docs_json})
        logging.info(f"LLM response: {response}")
        return JSONResponse(content={"response": response}, status_code=200)
    except Exception as e:
        logging.error(f"Error during simple search: {e}")
        return JSONResponse(
            content={"error": f"Error during simple search: {e}"},
            status_code=500
        )

@app.get("/list_documents")
async def list_documents_cdb():
    logging.info("Endpoint '/list_documents' called")
    try:
        store = get_store()
        docs = store._collection.get(include=["metadatas", "documents"])
        documents = []
        for metadata, content in zip(docs["metadatas"], docs["documents"]):
            documents.append({"content": content, "metadata": metadata})
        logging.info(f"Retrieved {len(documents)} documents from ChromaDB.")
        return JSONResponse(content={"documents": documents}, status_code=200)
    except Exception as e:
        logging.error(f"Error listing documents: {e}")
        return JSONResponse(content={"error": f"Error listing documents: {e}"}, status_code=500)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host="0.0.0.0", port=8080)