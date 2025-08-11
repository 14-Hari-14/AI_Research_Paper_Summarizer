import os
import fitz  # PyMuPDF
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles


from .image_utils import extract_and_cluster_images
from .llm_utils import get_llm, get_embeddings
from .models import QueryRequest, SummarizeRequest


from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="AI Research Paper Summarizer API")


UPLOAD_DIR = Path("uploads")
VECTOR_STORE_DIR = Path("backend/vector_stores")
STATIC_DIR = Path("backend/static")
UPLOAD_DIR.mkdir(exist_ok=True)
VECTOR_STORE_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)

# Mount the static directory to serve images
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# --- API Endpoints ---
@app.post("/upload-and-process/")
async def upload_and_process(api_key: str, file: UploadFile = File(...)):
    if not api_key:
        raise HTTPException(status_code=400, detail="Google AI API key is missing.")

    file_name_stem = Path(file.filename).stem
    file_path = UPLOAD_DIR / file.filename
    vector_store_path = VECTOR_STORE_DIR / f"{file_name_stem}.faiss"
    image_output_dir = STATIC_DIR / file_name_stem
    
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    try:
        image_filenames = extract_and_cluster_images(
            pdf_path=str(file_path),
            output_dir=image_output_dir
        )
        
        # Vector store creation remains the same
        if not vector_store_path.exists():
            loader = PyPDFLoader(str(file_path))
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
            texts = text_splitter.split_documents(documents)
            embeddings = get_embeddings(api_key)
            vector_store = FAISS.from_documents(texts, embeddings)
            vector_store.save_local(str(vector_store_path))
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process file: {str(e)}")
    finally:
        if file_path.exists():
            os.remove(file_path)
            
    return {
        "message": "File processed successfully.", 
        "file_name": file.filename, 
        "image_filenames": image_filenames
    }


# The /summarize/ and /chat/ endpoints remain exactly the same as the previous version.
# They are already designed to handle the data flow correctly.
@app.post("/summarize/")
async def summarize_section(api_key: str, request: SummarizeRequest):
    """Generates a summary for a specific section, now with a better prompt."""
    file_name_stem = Path(request.file_name).stem
    vector_store_path = Path("backend/vector_stores") / f"{file_name_stem}.faiss"
    
    if not vector_store_path.exists():
        raise HTTPException(status_code=404, detail="Vector store not found.")

    try:
        embeddings = get_embeddings(api_key)
        llm = get_llm(api_key)
        vector_store = FAISS.load_local(str(vector_store_path), embeddings, allow_dangerous_deserialization=True)
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 8})

        final_prompt_instruction = request.section_prompt

        # --- THIS IS THE KEY CHANGE ---
        # We make the instruction for the AI much more direct and provide a literal example.
        if request.image_filenames:
            image_list_str = ", ".join(request.image_filenames)
            final_prompt_instruction += (
                f"\n\nThis paper contains the following images: [{image_list_str}]. "
                "When your description refers to one of these images, you MUST end the paragraph with a clear reference in the format `(see: filename.png)`."
                "For example: 'This chart shows the performance increase. (see: figure_1.png)'"
            )

        prompt_template = f"""Based ONLY on the provided context, fulfill the following instruction.
        Instruction: {final_prompt_instruction}
        CONTEXT: {{context}}
        OUTPUT:"""
        PROMPT = PromptTemplate(template=prompt_template, input_variables=["context"])
        
        qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, chain_type_kwargs={"prompt": PROMPT})
        result = qa_chain.invoke({"query": "Generate the content for the section."})
        return {"summary": result["result"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/")
async def chat_with_doc(api_key: str, request: QueryRequest):
    # This function's code is correct and does not need to be changed.
    file_name_stem = Path(request.file_name).stem
    vector_store_path = VECTOR_STORE_DIR / f"{file_name_stem}.faiss"
    if not vector_store_path.exists():
        raise HTTPException(status_code=404, detail="Vector store not found.")
    try:
        embeddings = get_embeddings(api_key)
        llm = get_llm(api_key)
        vector_store = FAISS.load_local(str(vector_store_path), embeddings, allow_dangerous_deserialization=True)
        retriever = vector_store.as_retriever()
        prompt_template = "Answer the user's question based only on the provided context.\nContext: {context}\nQuestion: {question}"
        PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, chain_type_kwargs={"prompt": PROMPT})
        result = qa_chain.invoke({"query": request.query})
        return {"answer": result["result"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))