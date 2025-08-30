import os
import fitz
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
import re
import asyncio
import logging
import json
from aiolimiter import AsyncLimiter
from typing import List, Dict
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import markdown2
from xhtml2pdf import pisa
from io import BytesIO

from .image_utils import extract_and_cluster_images
from .llm_utils import get_llm, get_embeddings
from .models import QueryRequest, SummarizeRequest

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI(title="AI Research Paper Summarizer API")

# Directory setup
UPLOAD_DIR = Path("uploads")
VECTOR_STORE_DIR = Path("backend/vector_stores")
STATIC_DIR = Path("backend/static")
UPLOAD_DIR.mkdir(exist_ok=True)
VECTOR_STORE_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Rate limiter: 2 requests per 60 seconds
rate_limiter = AsyncLimiter(2, 60)

# Global request counter
request_counter = 0
MAX_DAILY_REQUESTS = 45  # Slightly below 50 to be safe

def ensure_figure_image_links(summary_text: str, figure_numbers: List[int], image_filenames: List[str]) -> str:
    """
    Ensures each figure section ends with a markdown image link if the image exists.
    """
    sections = re.split(r'\n\n(?=\*\*Figure \d+:\*\*)', summary_text)
    updated_sections = []
    valid_images = set(image_filenames)  # Only include existing images
    for section in sections:
        if not section.strip():
            continue
        for num in figure_numbers:
            figure_header = f"**Figure {num}:**"
            if section.startswith(figure_header):
                image_link = f"![](figure_{num}.png)"
                if f"figure_{num}.png" in valid_images and image_link not in section:
                    section = section.strip() + f"\n\n{image_link}"
                break
        updated_sections.append(section)
    return '\n\n'.join(updated_sections)

def convert_markdown_to_pdf(markdown_text: str, output_path: Path, image_dir: Path):
    """
    Converts markdown to PDF using markdown2 and xhtml2pdf.
    """
    try:
        # Convert markdown to HTML
        html = markdown2.markdown(markdown_text, extras=["tables", "fenced-code-blocks"])
        # Add CSS for basic styling
        html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ font-size: 24px; }}
                h3 {{ font-size: 18px; }}
                img {{ max-width: 100%; height: auto; }}
            </style>
        </head>
        <body>{html}</body>
        </html>
        """
        # Convert HTML to PDF
        with open(output_path, "wb") as pdf_file:
            pisa_status = pisa.CreatePDF(
                html,
                dest=pdf_file,
                path=str(image_dir),  # Base path for relative image links
            )
        if pisa_status.err:
            logger.error(f"Failed to convert markdown to PDF: {pisa_status.err}")
            return False
        logger.debug(f"PDF generated at {output_path}")
        return True
    except Exception as e:
        logger.error(f"Error converting markdown to PDF: {str(e)}")
        return False

async def async_retrieval_qa(llm, retriever, prompt: PromptTemplate, query: str, max_retries: int = 3) -> str:
    """Async wrapper for RetrievalQA with retry logic and rate limiting."""
    global request_counter
    for attempt in range(max_retries):
        if request_counter >= MAX_DAILY_REQUESTS:
            logger.error("Approaching daily API limit (50 requests). Stopping further requests.")
            raise HTTPException(status_code=429, detail="Daily API limit reached. Try again tomorrow or upgrade to a paid plan.")
        
        async with rate_limiter:
            try:
                logger.debug(f"Attempt {attempt + 1}/{max_retries} for query: {query}")
                request_counter += 1
                logger.debug(f"Total API requests made: {request_counter}")
                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=retriever,
                    chain_type_kwargs={"prompt": prompt}
                )
                result = await asyncio.to_thread(qa_chain.invoke, {"query": query})
                logger.debug(f"Query completed: {query}")
                return result["result"]
            except Exception as e:
                error_str = str(e).lower()
                retry_delay = 60
                if "rate_limit" in error_str or "429" in error_str:
                    try:
                        error_json = json.loads(str(e).split("For more information")[0] + "}")
                        retry_delay = error_json.get("retry_delay", {}).get("seconds", 60)
                    except json.JSONDecodeError:
                        retry_delay = 40 if "perminute" in error_str else 8
                    logger.warning(f"Rate limit hit for query {query}. Retrying after {retry_delay}s.")
                    await asyncio.sleep(retry_delay)
                elif "504" in error_str or "deadline exceeded" in error_str:
                    logger.warning(f"Timeout for query {query}. Retrying after 15s.")
                    await asyncio.sleep(15)
                else:
                    logger.error(f"Error processing query {query}: {str(e)}")
                    raise e
                if attempt == max_retries - 1:
                    logger.error(f"Max retries reached for query {query}: {str(e)}")
                    raise HTTPException(status_code=429, detail=f"Failed after {max_retries} retries: {str(e)}")

@app.post("/upload-and-process/")
async def upload_and_process(api_key: str, file: UploadFile = File(...)):
    global request_counter
    if not api_key:
        raise HTTPException(status_code=400, detail="Google AI API key is missing.")

    file_name_stem = Path(file.filename).stem
    file_path = UPLOAD_DIR / file.filename
    vector_store_path = VECTOR_STORE_DIR / f"{file_name_stem}.faiss"
    image_output_dir = STATIC_DIR / file_name_stem
    captions_path = STATIC_DIR / file_name_stem / "captions.json"
    
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    try:
        image_filenames = extract_and_cluster_images(
            pdf_path=str(file_path),
            output_dir=image_output_dir
        )
        
        # Extract figure captions using fitz
        doc = fitz.open(file_path)
        captions = {}
        for page_num in range(len(doc)):
            page = doc[page_num]
            d = page.get_text("dict")
            blocks = d["blocks"]
            img_blocks = [b for b in blocks if b["type"] == 1]  # Image blocks
            text_blocks = [b for b in blocks if b["type"] == 0]  # Text blocks
            
            for img_block in img_blocks:
                img_bbox = img_block["bbox"]
                nearby_texts = []
                for text_block in text_blocks:
                    text_bbox = text_block["bbox"]
                    if (text_bbox[1] > img_bbox[3] and abs(text_bbox[0] - img_bbox[0]) < 50) or \
                       (abs(text_bbox[1] - img_bbox[1]) < 50 and text_bbox[0] > img_bbox[2]):  # Below or beside
                        nearby_texts.append(" ".join(span["text"] for line in text_block["lines"] for span in line["spans"]))
                
                if nearby_texts:
                    caption_text = " ".join(nearby_texts)
                    logger.debug(f"Extracted caption text for image on page {page_num + 1}: {caption_text}")
                    match = re.search(r'[Ff]ig(?:ure)?\.?\s*(\d+)', caption_text, re.IGNORECASE)
                    if match:
                        fig_num = int(match.group(1))
                        captions[fig_num] = caption_text
                        logger.debug(f"Associated caption with Figure {fig_num}: {caption_text}")
                    else:
                        logger.warning(f"No figure number found in caption text: {caption_text}")
        
        # Save captions to JSON
        logger.debug(f"Saving captions to {captions_path}: {captions}")
        with open(captions_path, "w") as f:
            json.dump(captions, f)
        
        if not vector_store_path.exists():
            loader = PyPDFLoader(str(file_path))
            documents = loader.load()
            if len(documents) > 1000:
                logger.warning(f"Document {file.filename} has {len(documents)} pages, may cause timeouts.")
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
            texts = text_splitter.split_documents(documents)
            if len(texts) > 30:
                texts = texts[:30]
                logger.warning(f"Capped document chunks to {len(texts)} to avoid excessive API calls.")
            logger.debug(f"Split document into {len(texts)} chunks.")
            embeddings = get_embeddings(api_key)
            logger.debug(f"Generating embeddings for {len(texts)} chunks.")
            request_counter += len(texts)  # Approximate embedding API calls
            logger.debug(f"Total API requests after embeddings: {request_counter}")
            if request_counter >= MAX_DAILY_REQUESTS:
                raise HTTPException(status_code=429, detail="Daily API limit reached during embedding. Try again tomorrow or upgrade to a paid plan.")
            vector_store = FAISS.from_documents(texts, embeddings)
            vector_store.save_local(str(vector_store_path))
    
    except Exception as e:
        logger.error(f"Failed to process file {file.filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process file: {str(e)}")
    finally:
        if file_path.exists():
            os.remove(file_path)
            
    return {
        "message": "File processed successfully.", 
        "file_name": file.filename, 
        "image_filenames": image_filenames
    }

@app.post("/summarize/")
async def summarize_section(api_key: str, request: SummarizeRequest):
    global request_counter
    file_name_stem = Path(request.file_name).stem
    vector_store_path = VECTOR_STORE_DIR / f"{file_name_stem}.faiss"
    captions_path = STATIC_DIR / file_name_stem / "captions.json"
    md_path = STATIC_DIR / file_name_stem / "summary.md"
    pdf_path = STATIC_DIR / file_name_stem / "summary.pdf"
    
    if not vector_store_path.exists():
        raise HTTPException(status_code=404, detail="Vector store not found.")

    try:
        embeddings = get_embeddings(api_key)
        llm = get_llm(api_key)
        vector_store = FAISS.load_local(str(vector_store_path), embeddings, allow_dangerous_deserialization=True)
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

        # Load captions if available
        captions = {}
        if captions_path.exists():
            with open(captions_path, "r") as f:
                captions = json.load(f)
            logger.debug(f"Loaded captions: {captions}")
        else:
            logger.warning(f"No captions file found at {captions_path}")

        # For key visuals, retrieve all chunks to maximize figure coverage
        if "Describe the paper's most important visual elements" in request.section_prompt:
            docs = vector_store.similarity_search(" ", k=len(vector_store.docstore._dict))
            full_text = " ".join([doc.page_content for doc in docs])
            figure_numbers_found = re.findall(r'[Ff]ig(?:ure)?\.?\s*(\d+)', full_text, re.IGNORECASE)
            if not figure_numbers_found:
                return {
                    "summary": "No figures were explicitly mentioned in the document.",
                    "markdown_url": "",
                    "pdf_url": ""
                }
            unique_figure_numbers = sorted(list(set(map(int, figure_numbers_found))))
            logger.debug(f"Found figures: {unique_figure_numbers}")
            
            # Prepare async tasks for figure descriptions
            tasks = []
            for num in unique_figure_numbers:
                if request_counter >= MAX_DAILY_REQUESTS:
                    logger.error("Daily API limit reached. Stopping figure processing.")
                    return {
                        "summary": "Daily API limit exceeded. Please try again tomorrow or upgrade to a paid plan.",
                        "markdown_url": "",
                        "pdf_url": ""
                    }
                image_list_str = ", ".join(request.image_filenames or [])
                caption = captions.get(str(num), f"Figure {num}: No caption available")
                desc_prompt_template = f"""Provide a detailed description of Figure {num} based on the provided context and figure caption. 
If the context lacks specific details about Figure {num} or is empty, generate a detailed explanation by expanding on the figure caption alone, inferring its content and significance based on the caption and the general context of the document (e.g., results, methodology, or conclusions). Ensure the description is informative and relevant to the likely role of the figure in the paper.

Context: {{context}}

Figure Caption: {caption}

The available images are [{image_list_str}]."""
                desc_prompt = PromptTemplate(template=desc_prompt_template, input_variables=["context"])
                tasks.append(async_retrieval_qa(
                    llm, retriever, desc_prompt, f"Describe the content and significance of Figure {num}."
                ))

            # Run tasks in parallel with rate limiting
            descriptions = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Format results
            formatted_descriptions = []
            for num, desc in zip(unique_figure_numbers, descriptions):
                if isinstance(desc, Exception):
                    formatted_descriptions.append(f"**Figure {num}:** Error: {str(desc)}")
                    logger.error(f"Error describing Figure {num}: {str(desc)}")
                else:
                    desc = desc.strip()
                    formatted_descriptions.append(f"**Figure {num}:** {desc}")
            raw_summary = "\n\n".join(formatted_descriptions)
            final_summary = ensure_figure_image_links(raw_summary, unique_figure_numbers, request.image_filenames)

            # Save markdown file
            md_path.parent.mkdir(exist_ok=True)
            with open(md_path, "w") as f:
                f.write(final_summary)
            logger.debug(f"Saved markdown to {md_path}")

            # Convert to PDF
            pdf_success = convert_markdown_to_pdf(final_summary, pdf_path, STATIC_DIR / file_name_stem)
            pdf_url = f"/static/{file_name_stem}/summary.pdf" if pdf_success else ""

            return {
                "summary": final_summary,
                "markdown_url": f"/static/{file_name_stem}/summary.md",
                "pdf_url": pdf_url
            }

        # Standard summarization for other sections
        else:
            if request_counter >= MAX_DAILY_REQUESTS:
                logger.error("Daily API limit reached. Stopping summarization.")
                return {
                    "summary": "Daily API limit exceeded. Please try again tomorrow or upgrade to a paid plan.",
                    "markdown_url": "",
                    "pdf_url": ""
                }
            final_prompt_instruction = request.section_prompt
            if request.image_filenames:
                image_list_str = ", ".join(request.image_filenames)
                final_prompt_instruction += (
                    f"\n\nThis paper contains the following images: [{image_list_str}]. "
                    "When your description refers to one of these images, end the paragraph with `(see: filename.png)`."
                )
            prompt_template = f"""Based ONLY on the provided context, fulfill the following instruction.
Instruction: {final_prompt_instruction}
CONTEXT: {{context}}
OUTPUT:"""
            PROMPT = PromptTemplate(template=prompt_template, input_variables=["context"])
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                chain_type_kwargs={"prompt": PROMPT}
            )
            result = await asyncio.to_thread(qa_chain.invoke, {"query": "Generate the content for the section."})
            final_summary = result["result"]

            # Save markdown file
            md_path.parent.mkdir(exist_ok=True)
            with open(md_path, "w") as f:
                f.write(final_summary)
            logger.debug(f"Saved markdown to {md_path}")

            # Convert to PDF
            pdf_success = convert_markdown_to_pdf(final_summary, pdf_path, STATIC_DIR / file_name_stem)
            pdf_url = f"/static/{file_name_stem}/summary.pdf" if pdf_success else ""

            return {
                "summary": final_summary,
                "markdown_url": f"/static/{file_name_stem}/summary.md",
                "pdf_url": pdf_url
            }
            
    except Exception as e:
        logger.error(f"Failed to summarize {request.file_name}: {str(e)}")
        if "429" in str(e).lower():
            return {
                "summary": "Daily API limit. Please try again tomorrow or upgrade to a paid plan.",
                "markdown_url": "",
                "pdf_url": ""
            }
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/")
async def chat_with_doc(api_key: str, request: QueryRequest):
    global request_counter
    file_name_stem = Path(request.file_name).stem
    vector_store_path = VECTOR_STORE_DIR / f"{file_name_stem}.faiss"
    if not vector_store_path.exists():
        raise HTTPException(status_code=404, detail="Vector store not found.")
    try:
        if request_counter >= MAX_DAILY_REQUESTS:
            logger.error("Daily API limit reached. Stopping chat.")
            return {"answer": "Daily API limit exceeded. Please try again tomorrow or upgrade to a paid plan."}
        embeddings = get_embeddings(api_key)
        llm = get_llm(api_key)
        vector_store = FAISS.load_local(str(vector_store_path), embeddings, allow_dangerous_deserialization=True)
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 8})
        prompt_template = "Answer the user's question based only on the provided context.\nContext: {context}\nQuestion: {question}"
        PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": PROMPT}
        )
        result = await asyncio.to_thread(qa_chain.invoke, {"query": request.query})
        return {"answer": result["result"]}
    except Exception as e:
        logger.error(f"Failed to process chat for {request.file_name}: {str(e)}")
        if "429" in str(e).lower():
            return {"answer": "Daily API limit. Please try again tomorrow or upgrade to a paid plan."}
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/debug-request-count/")
async def debug_request_count():
    """Debug endpoint to check current API request count."""
    global request_counter
    return {"total_requests_made": request_counter, "max_daily_requests": MAX_DAILY_REQUESTS}