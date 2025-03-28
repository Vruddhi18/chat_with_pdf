import pdfplumber
import pytesseract
import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF (uses OCR if needed)."""
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"

    if not text.strip():
        doc = fitz.open(pdf_path)
        for page in doc:
            pix = page.get_pixmap()
            text += pytesseract.image_to_string(pix.tobytes(), lang="eng") + "\n"

    return text
