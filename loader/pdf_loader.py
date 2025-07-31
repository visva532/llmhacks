import fitz  # PyMuPDF
import os

def extract_text_from_pdf(file_path):
    pages = []
    with fitz.open(file_path) as doc:
        for i, page in enumerate(doc):
            text = page.get_text().strip()
            if text:  # Skip empty pages
                pages.append({
                    "page_number": i + 1,
                    "text": text,
                    "source": os.path.basename(file_path)
                })
    return pages
