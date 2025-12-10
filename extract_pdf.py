from pypdf import PdfReader

reader = PdfReader("2025_C1_coursework_final.pdf")
with open("coursework_requirements.txt", "w") as f:
    for page in reader.pages:
        f.write(page.extract_text())
        f.write("\n")
