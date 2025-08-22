# AI Paper Summarizer

A web application that summarizes research papers and extracts key diagrams using Retrieval-Augmented Generation (RAG) and Google Gemini LLM.  
It supports PDF upload, image extraction, section-wise summarization, and interactive chat with the document.

---

## Examples

## Features

- **PDF Upload:** Upload research papers for processing.
- **Image Extraction:** Automatically extracts and saves figures/diagrams from the PDF.
- **Section Summarization:** Summarizes sections of the paper, referencing extracted images.
- **Interactive Chat:** Ask questions about the paper and get context-aware answers.
- **FastAPI Backend:** Handles all processing, RAG pipeline, and serves images.
- **Frontend App:** Simple interface for interacting with the backend.

---

## Folder Structure

```
AI_Paper_Summarizer/
├── backend/
│   ├── main.py                # FastAPI app and endpoints
│   ├── image_utils.py         # Image extraction utilities
│   ├── llm_utils.py           # LLM and embedding helpers
│   ├── models.py              # Pydantic models
│   ├── static/                # Extracted images (per paper)
│   ├── vector_stores/         # FAISS vector stores (per paper)
│   └── __pycache__/
├── frontend/
│   └── app.py                 # Frontend application (entry point)
├── .env                       # API keys and environment variables
```

---

## Setup

### 1. Clone the Repository

```sh
git clone https://github.com/yourusername/AI_Paper_Summarizer.git
cd AI_Paper_Summarizer
```

### 2. Install Dependencies

```sh
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Set Up API Keys

- Add your Google Gemini API key to the `.env` file in the project root:

  ```
  GEMINI_API_KEY=your_google_gemini_api_key
  ```

---

## Usage

### 1. Start the Backend

```sh
cd backend
uvicorn main:app --reload
```

### 2. Start the Frontend

```sh
cd ../frontend
python app.py
```

### 3. Open the App

- Access the frontend in your browser (URL depends on your frontend framework).
- Upload a PDF, view extracted diagrams, summarize sections, and chat with the paper.

---

## How It Works

- **Upload:** PDF is uploaded and saved.
- **Image Extraction:** Figures are extracted and saved in `backend/static/<paper_name>/`.
- **Chunking & Embedding:** Text is chunked and embedded; stored in FAISS vector store.
- **Summarization:** RAG pipeline retrieves relevant chunks and generates summaries, referencing images.
- **Chat:** Ask questions; answers are generated using retrieved context from the paper.

---

## Customization

- **Chunk Size/Overlap:** Tune in `main.py` for best results.
- **Prompt Engineering:** Adjust prompts for better diagram coverage.
- **Image Extraction:** Improve `image_utils.py` for more accurate caption extraction.

---

## Technologies Used

- **Python 3.10+**
- **FastAPI** (backend API)
- **Google Gemini API** (LLM for summarization and chat)
- **LangChain** (RAG pipeline and embeddings)
- **FAISS** (vector store for document retrieval)
- **PyMuPDF** (PDF and image extraction)
- **Streamlit** or **Gradio** (frontend interface)
- **Pydantic** (data validation)
- **dotenv** (environment variable management)

## License

MIT License

---

## Credits

- [LangChain](https://github.com/langchain-ai/langchain)
- [Google Gemini](https://ai.google.dev/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [PyMuPDF](https://pymupdf.readthedocs.io/en/latest/)
