from utils.resume_parser import extract_text

text = extract_text("resumes/sample.pdf")

print(text[:500])