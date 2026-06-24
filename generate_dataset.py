"""
Script to generate candidate_resume_dataset.csv for AI-Hiring Assistant.
Run once: python generate_dataset.py
"""
import pandas as pd
import random
import os

random.seed(42)

ROLES = [
    "Data Scientist", "ML Engineer", "Software Engineer", "Data Analyst",
    "NLP Engineer", "Backend Developer", "DevOps Engineer", "AI Researcher",
    "Full Stack Developer", "Cloud Architect", "Computer Vision Engineer",
    "Business Intelligence Analyst", "Data Engineer", "Python Developer",
    "Deep Learning Researcher",
]

EDUCATIONS = [
    "B.Tech Computer Science", "M.Tech AI/ML", "B.E. Information Technology",
    "M.Sc Data Science", "B.Sc Computer Science", "MBA Analytics",
    "PhD Machine Learning", "B.Tech Electronics", "M.Tech Software Engineering",
    "B.Tech AI & Data Science",
]

LOCATIONS = [
    "Bangalore", "Hyderabad", "Mumbai", "Delhi", "Chennai",
    "Pune", "Kolkata", "Ahmedabad", "Noida", "Gurgaon",
    "Remote", "San Francisco", "London", "Singapore", "Dubai",
]

SKILLS_POOL = [
    "Python", "Machine Learning", "Deep Learning", "NLP", "SQL",
    "TensorFlow", "PyTorch", "Pandas", "NumPy", "Scikit-learn",
    "React", "JavaScript", "TypeScript", "Node.js", "Docker",
    "Kubernetes", "AWS", "Azure", "GCP", "Git",
    "FastAPI", "Flask", "Django", "REST API", "GraphQL",
    "Data Analysis", "Data Visualization", "Power BI", "Tableau",
    "Communication", "Leadership", "Problem Solving", "Agile", "Scrum",
    "Java", "C++", "Spark", "Hadoop", "Kafka",
    "Computer Vision", "Reinforcement Learning", "Statistics",
    "System Design", "Microservices", "CI/CD", "DevOps", "Airflow",
]

FIRST_NAMES = [
    "Aarav","Aditi","Akash","Ananya","Arjun","Bhavna","Chirag","Deepa",
    "Farhan","Gaurav","Harsh","Ishaan","Jaya","Karan","Lavanya","Manish",
    "Neha","Omkar","Priya","Rahul","Sakshi","Tanvi","Uday","Vijay","Yash",
    "Zara","Riya","Siddharth","Meera","Rohit","Aisha","Dev","Pooja","Nikhil",
    "Simran","Aditya","Kavya","Rohan","Shruti","Vikram","Divya","Suresh",
    "Ankita","Rajesh","Sneha","Amit","Pallavi","Kunal","Tara","Vivek",
]

LAST_NAMES = [
    "Sharma","Patel","Singh","Kumar","Verma","Gupta","Mehta","Reddy",
    "Nair","Joshi","Shah","Iyer","Chopra","Malhotra","Kapoor","Bose",
    "Das","Roy","Chatterjee","Mukherjee","Pillai","Menon","Agarwal","Saxena",
    "Pandey","Tiwari","Mishra","Shukla","Yadav","Trivedi","Desai","Rao",
]

RESUME_TEMPLATES = [
    "Experienced {role} with {exp} years of expertise in {skills}. Passionate about building scalable systems and solving complex problems. Strong background in {edu}.",
    "Results-driven {role} with {exp}+ years in the industry. Proficient in {skills}. Delivered multiple high-impact projects across domains.",
    "Dynamic {role} skilled in {skills}. {exp} years of hands-on experience. Graduated from {edu} program. Looking for challenging opportunities.",
    "Senior {role} with deep expertise in {skills}. {exp} years of professional experience. Committed to data-driven decision making.",
    "Motivated {role} | {exp} years exp | Key skills: {skills}. Education: {edu}. Strong analytical and communication skills.",
]

def make_email(first, last):
    domains = ["gmail.com","yahoo.com","outlook.com","hotmail.com","company.com"]
    return f"{first.lower()}.{last.lower()}{random.randint(10,99)}@{random.choice(domains)}"

rows = []
for i in range(1, 201):
    first    = random.choice(FIRST_NAMES)
    last     = random.choice(LAST_NAMES)
    name     = f"{first} {last}"
    role     = random.choice(ROLES)
    exp      = round(random.uniform(0.5, 15.0), 1)
    edu      = random.choice(EDUCATIONS)
    loc      = random.choice(LOCATIONS)
    email    = make_email(first, last)
    phone    = f"+91-{random.randint(7000000000, 9999999999)}"

    # Skills: role-biased + random
    n_skills = random.randint(4, 12)
    skills   = random.sample(SKILLS_POOL, min(n_skills, len(SKILLS_POOL)))

    tmpl     = random.choice(RESUME_TEMPLATES)
    resume   = tmpl.format(
        role=role, exp=int(exp),
        skills=", ".join(skills[:5]),
        edu=edu,
    )

    rows.append({
        "Candidate_ID":      i,
        "Candidate_Name":    name,
        "Email":             email,
        "Phone":             phone,
        "Current_Role":      role,
        "Experience_Years":  exp,
        "Education":         edu,
        "Location":          loc,
        "Skills":            ", ".join(skills),
        "Resume_Text":       resume,
    })

df = pd.DataFrame(rows)
os.makedirs("data", exist_ok=True)
df.to_csv("data/candidate_resume_dataset.csv", index=False)
print(f"✅ Generated {len(df)} candidates → data/candidate_resume_dataset.csv")
