import pdfplumber

def extract_resume_text(uploaded_resume):
    text = ""

    with pdfplumber.open(uploaded_resume) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

    return text