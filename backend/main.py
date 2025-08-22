import os
import fitz  # PyMuPDF
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
import re


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
    file_name_stem = Path(request.file_name).stem
    vector_store_path = Path("backend/vector_stores") / f"{file_name_stem}.faiss"
    
    if not vector_store_path.exists():
        raise HTTPException(status_code=404, detail="Vector store not found.")

    try:
        embeddings = get_embeddings(api_key)
        llm = get_llm(api_key)
        
        # Load the simple FAISS vector store
        vector_store = FAISS.load_local(str(vector_store_path), embeddings, allow_dangerous_deserialization=True)
        # Create the standard retriever
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 8})

        # --- NEW HYBRID LOGIC FOR THE DIAGRAMS SECTION ---
        if "Describe the paper's most important visual elements" in request.section_prompt:
            
            # Step A: Reconstruct the full text by fetching all chunks from the vector store.
            # We use a dummy query and a large 'k' to get everything.
            docs = vector_store.similarity_search(" ", k=len(vector_store.docstore._dict))
            full_text = " ".join([doc.page_content for doc in docs])
            
            # Step B: Deterministically find all figure numbers using regex.
            figure_numbers_found = re.findall(r'[Ff]ig(?:ure)?\.?\s*(\d+)', full_text)
            if not figure_numbers_found:
                return {"summary": "No figures were explicitly mentioned in the document."}
            unique_figure_numbers = sorted(list(set(map(int, figure_numbers_found))))
            
            # Step C: Run a targeted RAG query for each identified figure.
            descriptions = []
            for num in unique_figure_numbers:
                image_list_str = ", ".join(request.image_filenames or [])
                
                desc_prompt_template = f"""Based ONLY on the provided context, provide a detailed description of Figure {num}.
                The available images are [{image_list_str}]. If the context mentions an associated image filename like 'figure_{num}.png', you MUST end the paragraph with a clear reference in the format `(see: figure_{num}.png)`.
                CONTEXT: {{context}}
                OUTPUT:"""
                
                desc_prompt = PromptTemplate(template=desc_prompt_template, input_variables=["context"])
                desc_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, chain_type_kwargs={"prompt": desc_prompt})
                description = desc_chain.invoke({"query": f"Find and describe the content and significance of Figure {num}."})["result"]
                descriptions.append(f"**Figure {num}:** {description}")

            return {"summary": "\n\n".join(descriptions)}

        # --- ORIGINAL LOGIC FOR ALL OTHER TEXT-BASED SECTIONS ---
        else:
            final_prompt_instruction = request.section_prompt
            # This part is your original, working code for text sections.
            if request.image_filenames:
                image_list_str = ", ".join(request.image_filenames)
                final_prompt_instruction += (
                    f"\n\nThis paper contains the following images: [{image_list_str}]. "
                    "When your description refers to one of these images, you MUST end the paragraph with a clear reference in the format `(see: filename.png)`."
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