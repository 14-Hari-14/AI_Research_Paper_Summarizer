from PyPDF2 import PdfReader as pfr
import fitz  
import os

class DataExtractor:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path

    def extract_text_from_pdf(self):
        """Extracts all text from the PDF."""
        try:
            reader = pfr(self.pdf_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            print(f"Error reading PDF with PyPDF2: {e}")
            return ""

    def extract_images_from_pdf(self, output_dir="output_images"):
        """Extracts images and returns an ordered list of their paths."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        pdf = fitz.open(self.pdf_path)
        image_paths = []
        image_count = 0

        for page_num in range(len(pdf)):
            for img_info in pdf[page_num].get_images(full=True):
                image_count += 1
                xref = img_info[0]
                base_image = pdf.extract_image(xref)
                image_bytes = base_image["image"]
                
                # Use a simple, ordered filename
                image_filename = os.path.join(output_dir, f"figure_{image_count}.png")
                
                with open(image_filename, "wb") as f:
                    f.write(image_bytes)
                image_paths.append(image_filename)
        
        print(f"Extracted {len(image_paths)} images.")
        return image_paths