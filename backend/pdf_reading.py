import pdfplumber
from pathlib import Path
import json

def content_extraction(pdf_path: Path, store_extraction: bool = False, log_path: Path = None) -> str:
    """
    Extract text content from a PDF file.

    Args:
        pdf_path: Path to the PDF file.
        store_extraction: If True, saves the extracted content as a JSON file at output_path.
        log_path: Directory to save the JSON file. Required if store_extraction is True.

    Returns:
        A single string with all page text joined by double newlines.
    """
    content = ""
    pages = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            pages.append(text)
        
        content = "\n\n".join(pages)
    
    title = pdf_path.stem
    if store_extraction:
        note = {}
        note['title'] = title
        note['content'] = content

        log_path.mkdir(parents=True, exist_ok=True)
        output_file = log_path / (title + '.json')
        with open(output_file, "w") as f:
            json.dump(note, f, indent=4)
    
    return content