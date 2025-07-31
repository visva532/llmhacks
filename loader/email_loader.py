import extract_msg
import os

def extract_text_from_email(file_path):
    """
    Extracts subject and body from a .msg email file.
    Returns a list with one dict containing 'page_number', 'text', and 'source'.
    """
    if not file_path.endswith(".msg"):
        raise ValueError(f"Unsupported file type: {file_path}")

    msg = extract_msg.Message(file_path)
    msg_subject = msg.subject or "No Subject"
    msg_body = msg.body or "No Body"

    return [{
        "page_number": 1,  # To maintain consistency with .pdf and .docx
        "text": f"Subject: {msg_subject}\n\n{msg_body}",
        "source": os.path.basename(file_path)
    }]
