from docx import Document
import os

def extract_text_from_docx(file_path):
    """
    Extracts clean text from a .docx file and returns a list with one entry.
    Maintains structure compatibility with PDF loader by including 'page_number'.
    """
    if not file_path.endswith(".docx"):
        raise ValueError(f"Unsupported file type: {file_path}")

    doc = Document(file_path)
    full_text = [para.text.strip() for para in doc.paragraphs if para.text.strip()]

    return [{
        "page_number": 1,  # Use 1 for consistency
        "text": "\n".join(full_text),
        "source": os.path.basename(file_path)
    }]
