from PyPDF2 import PdfReader as pfr

class TextExtractor:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
    
    def extract_text_from_pdf(self):
        reader = pfr(self.pdf_path)
        size = len(reader.pages)
        text = ""
        for i in range(size):
            page = reader.pages[i]
            text += page.extract_text() + "\n"
        
        return text
