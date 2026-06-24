skills_db = [
    "Python",
    "Machine Learning",
    "Deep Learning",
    "NLP",
    "SQL",
    "TensorFlow",
    "PyTorch",
    "Pandas",
    "NumPy",
    "Scikit-learn"
]

def extract_skills(text):

    found_skills = []

    for skill in skills_db:
        if skill.lower() in text.lower():
            found_skills.append(skill)

    return found_skills