#!/usr/bin/env python3
"""Extract text from PDF file"""

import sys

try:
    import PyPDF2
except ImportError:
    print("Installing PyPDF2...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "PyPDF2"])
    import PyPDF2

def extract_pdf_text(pdf_path):
    """Extract all text from PDF"""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page_num, page in enumerate(reader.pages):
            text += f"\n--- Page {page_num + 1} ---\n"
            text += page.extract_text()
    return text

if __name__ == "__main__":
    pdf_path = "/Users/rainiver/Desktop/work related/DIS/C1/c1_coursework/2025_C1_coursework_final.pdf"
    text = extract_pdf_text(pdf_path)
    print(text)
