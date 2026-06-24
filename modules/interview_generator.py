"""
modules/interview_generator.py
AI Interview Question Generator
Generates role-specific, skill-aware questions without requiring an API key.
Falls back to a rich built-in question bank when no API is configured.
"""

import random
from typing import Optional

# ── Built-in question bank ──────────────────────────────────────────────────
# Structure: {role_keyword: {category: {difficulty: [questions]}}}

QUESTION_BANK = {
    # ── Data Scientist ──────────────────────────────────────────────────────
    "data scientist": {
        "Technical": {
            "Easy": [
                "What is the difference between supervised and unsupervised learning?",
                "Explain what a p-value means in hypothesis testing.",
                "What is the purpose of train/test split in machine learning?",
                "Define precision and recall. When would you use each?",
                "What is overfitting and how do you prevent it?",
            ],
            "Medium": [
                "Explain the bias-variance tradeoff with an example.",
                "How does a Random Forest differ from a single Decision Tree?",
                "What is regularization? Compare L1 (Lasso) and L2 (Ridge).",
                "Describe the steps of a typical data science pipeline.",
                "How would you handle a highly imbalanced dataset?",
            ],
            "Hard": [
                "Explain the difference between Random Forest and XGBoost. When would you choose one over the other?",
                "Walk me through how you would build an end-to-end ML pipeline for a fraud detection system.",
                "How do you interpret SHAP values and when are they preferable to LIME?",
                "Describe how gradient boosting works mathematically.",
                "How would you design an A/B test for a new recommendation algorithm?",
            ],
        },
        "Behavioral": {
            "Easy": [
                "Tell me about a data project you're most proud of.",
                "How do you stay updated with the latest developments in data science?",
            ],
            "Medium": [
                "Describe a time when your data analysis led to a significant business decision.",
                "Tell me about a situation where the data contradicted your initial hypothesis.",
            ],
            "Hard": [
                "Describe a complex analytical problem you solved. What was your thought process?",
                "Tell me about a time you had to convince stakeholders to change their decision based on data.",
            ],
        },
        "Problem Solving": {
            "Easy": [
                "How would you find the top 3 most frequent values in a pandas Series?",
                "How do you detect and handle missing values in a dataset?",
            ],
            "Medium": [
                "You have a dataset with 1 million rows and 500 features. How would you approach feature selection?",
                "How would you detect and remove outliers from a numerical column?",
            ],
            "Hard": [
                "Design a real-time anomaly detection system for financial transactions. What architecture would you use?",
                "You notice your model performs well on training data but poorly in production. Walk me through your debugging process.",
            ],
        },
        "Scenario": {
            "Medium": [
                "A stakeholder asks you to predict next quarter's revenue with 95% accuracy. How do you respond?",
                "Your model's AUC suddenly drops from 0.92 to 0.74 in production. What do you do?",
            ],
            "Hard": [
                "You're asked to reduce customer churn by 20% in 3 months. Describe your entire approach.",
                "How would you build a recommendation system for an e-commerce platform from scratch?",
            ],
        },
    },

    # ── ML Engineer ─────────────────────────────────────────────────────────
    "ml engineer": {
        "Technical": {
            "Easy": [
                "What is the difference between a model's accuracy and its F1 score?",
                "Explain what batch normalization does in a neural network.",
                "What is transfer learning and why is it useful?",
                "What is the role of a loss function in training a neural network?",
                "What is the vanishing gradient problem?",
            ],
            "Medium": [
                "Explain how backpropagation works.",
                "What is the difference between CNN and RNN? When would you use each?",
                "How do you optimize hyperparameters in a machine learning model?",
                "What is MLflow and how does it help in ML experiment tracking?",
                "Describe the differences between online learning and batch learning.",
            ],
            "Hard": [
                "How would you deploy a TensorFlow model as a REST API at scale?",
                "Explain how you would build a model monitoring system to detect data drift.",
                "Compare ONNX, TorchScript, and TensorFlow SavedModel for model serialization.",
                "How would you implement a feature store? What are the key design considerations?",
                "Describe an architecture for training large language models efficiently.",
            ],
        },
        "Technical (MLOps)": {
            "Medium": [
                "What CI/CD practices do you apply to ML pipelines?",
                "How do you version datasets and models in production?",
            ],
            "Hard": [
                "Design an MLOps pipeline for a model that retrains weekly on new data.",
                "How would you implement blue-green deployment for an ML model?",
            ],
        },
        "Problem Solving": {
            "Medium": [
                "Your model inference latency is 800ms. The requirement is 100ms. How do you optimize it?",
                "How do you handle concept drift in a deployed model?",
            ],
            "Hard": [
                "Design a system that serves 10,000 predictions per second with 99.9% uptime.",
                "How would you implement a distributed training job for a 7B parameter model?",
            ],
        },
        "Behavioral": {
            "Medium": [
                "Tell me about a production model failure you experienced. What happened and what did you learn?",
                "How do you balance model accuracy with inference speed?",
            ],
            "Hard": [
                "Describe the most complex ML system you've built. What were the key architectural decisions?",
            ],
        },
    },

    # ── Software Engineer ────────────────────────────────────────────────────
    "software engineer": {
        "Technical": {
            "Easy": [
                "What is the difference between a list and a tuple in Python?",
                "Explain the concept of REST APIs.",
                "What is the difference between GET and POST HTTP methods?",
                "What is version control and why is it important?",
                "Explain object-oriented programming principles (SOLID).",
            ],
            "Medium": [
                "What is the difference between SQL and NoSQL databases?",
                "Explain the concept of microservices vs monolithic architecture.",
                "How does garbage collection work in Python?",
                "What is a decorator in Python? Write an example.",
                "Explain the concept of database indexing and when to use it.",
            ],
            "Hard": [
                "Design a URL shortening service like bit.ly. Cover the architecture, database, and scalability.",
                "How would you design a distributed cache? What consistency guarantees would you provide?",
                "Explain the CAP theorem and give real-world examples.",
                "How does async/await work in Python? Explain with an event loop example.",
                "Design a rate limiter for a public API. What algorithms would you consider?",
            ],
        },
        "Problem Solving": {
            "Easy": [
                "Write a function to check if a string is a palindrome.",
                "How would you reverse a linked list?",
            ],
            "Medium": [
                "Given an array, find two numbers that sum to a target. What is the optimal complexity?",
                "Implement a LRU cache in Python.",
            ],
            "Hard": [
                "Design a thread-safe singleton class in Python.",
                "How would you implement a job queue with retry logic and dead-letter queues?",
            ],
        },
        "Behavioral": {
            "Medium": [
                "Tell me about a time you had to refactor a large codebase. What was your approach?",
                "Describe a situation where you disagreed with a technical decision and how you handled it.",
            ],
            "Hard": [
                "What's the largest scale system you've worked on? What were the biggest challenges?",
            ],
        },
    },

    # ── Data Analyst ─────────────────────────────────────────────────────────
    "data analyst": {
        "Technical": {
            "Easy": [
                "What is the difference between INNER JOIN and LEFT JOIN in SQL?",
                "How do you handle duplicate records in a dataset?",
                "What are the main chart types and when would you use each?",
                "Explain what a pivot table is.",
                "What is the difference between COUNT(*) and COUNT(column)?",
            ],
            "Medium": [
                "Write a SQL query to find the top 5 customers by total revenue.",
                "How do you perform cohort analysis?",
                "Explain window functions in SQL with an example.",
                "What is the difference between OLAP and OLTP?",
                "How would you calculate customer lifetime value (CLV)?",
            ],
            "Hard": [
                "Design a dashboard to track e-commerce KPIs. What metrics would you include and why?",
                "How would you identify seasonality in a sales dataset?",
                "Describe how you would build a customer segmentation model using SQL and Python.",
            ],
        },
        "Behavioral": {
            "Medium": [
                "Tell me about an insight you discovered that had a significant business impact.",
                "Describe a time you had to explain complex data findings to a non-technical audience.",
            ],
        },
    },

    # ── Python Developer ──────────────────────────────────────────────────────
    "python developer": {
        "Technical": {
            "Easy": [
                "What are Python's mutable and immutable data types?",
                "Explain list comprehensions with an example.",
                "What is the GIL (Global Interpreter Lock)?",
                "What is the difference between *args and **kwargs?",
                "How does Python's memory management work?",
            ],
            "Medium": [
                "Explain Python generators and when to use them over lists.",
                "What is the difference between @staticmethod and @classmethod?",
                "How do you implement context managers in Python?",
                "Explain how Python's asyncio event loop works.",
                "What is metaclass in Python?",
            ],
            "Hard": [
                "How would you optimize a Python function that processes 10 million records?",
                "Explain Python's descriptor protocol and how it powers properties.",
                "Design a plugin architecture in Python using abstract base classes.",
                "How do you profile and debug memory leaks in a Python application?",
            ],
        },
        "Problem Solving": {
            "Medium": [
                "Write a Python function to flatten a nested dictionary.",
                "Implement a thread-safe counter in Python.",
            ],
            "Hard": [
                "Write a Python script to parse and validate large JSON files efficiently.",
            ],
        },
    },

    # ── AI Engineer ───────────────────────────────────────────────────────────
    "ai engineer": {
        "Technical": {
            "Easy": [
                "What is the difference between narrow AI and general AI?",
                "Explain what a transformer model is at a high level.",
                "What is fine-tuning a pre-trained model?",
                "What is prompt engineering?",
                "What is the difference between zero-shot and few-shot learning?",
            ],
            "Medium": [
                "Explain the attention mechanism in transformers.",
                "What is RAG (Retrieval-Augmented Generation)?",
                "How do you evaluate an LLM's output quality?",
                "What is the difference between BERT and GPT architectures?",
                "Explain LangChain and when you would use it.",
            ],
            "Hard": [
                "Design an enterprise AI assistant using RAG with a vector database. Cover the architecture, chunking strategy, and retrieval logic.",
                "How would you implement a feedback loop to continuously improve an LLM application?",
                "Explain how LoRA fine-tuning works and when to use it vs full fine-tuning.",
                "How do you mitigate hallucinations in an LLM-based production system?",
            ],
        },
        "Scenario": {
            "Medium": [
                "A client wants to use GPT-4 to automate customer support. What risks do you identify?",
            ],
            "Hard": [
                "Design an AI system that can answer questions about a 10,000-page legal document database.",
            ],
        },
    },
}

# ── Generic fallback bank (role-agnostic) ────────────────────────────────────
GENERIC_QUESTIONS = {
    "Behavioral": {
        "Easy": [
            "Tell me about yourself and your career journey.",
            "What motivated you to apply for this role?",
            "What are your greatest strengths?",
        ],
        "Medium": [
            "Describe a challenging project and how you overcame obstacles.",
            "Tell me about a time you worked in a team under a tight deadline.",
            "How do you prioritize tasks when you have multiple deadlines?",
        ],
        "Hard": [
            "Tell me about a time you failed. What did you learn?",
            "Describe a situation where you had to lead without formal authority.",
            "How have you handled a major conflict with a colleague or manager?",
        ],
    },
    "Problem Solving": {
        "Easy": [
            "How do you approach debugging a problem you've never seen before?",
        ],
        "Medium": [
            "Walk me through how you would break down a large, ambiguous project.",
        ],
        "Hard": [
            "You are given a critical system outage at 2 AM. What is your process?",
        ],
    },
}

# ── Skill-specific question inserts ──────────────────────────────────────────
SKILL_QUESTIONS = {
    "Python":             "Explain how Python's asyncio module differs from threading.",
    "Machine Learning":   "What is the difference between bagging and boosting?",
    "Deep Learning":      "Explain the concept of residual connections in deep networks.",
    "NLP":                "What is the difference between stemming and lemmatization?",
    "SQL":                "Write a SQL query to find employees who earn more than their manager.",
    "TensorFlow":         "How do you save and restore a TensorFlow model?",
    "PyTorch":            "What is the difference between torch.no_grad() and torch.detach()?",
    "AWS":                "What is the difference between EC2, ECS, and Lambda?",
    "Docker":             "Explain the difference between Docker image and Docker container.",
    "Kubernetes":         "What is the role of a Kubernetes pod and how does it differ from a container?",
    "FastAPI":            "How does FastAPI handle async endpoints? What are the performance implications?",
    "Pandas":             "How would you efficiently merge two DataFrames with 10M rows each?",
    "Scikit-learn":       "Explain the purpose of sklearn Pipeline and why it prevents data leakage.",
    "Spark":              "What is the difference between Spark RDD, DataFrame, and Dataset?",
    "Git":                "Explain Git rebase vs merge. When would you use each?",
    "DevOps":             "What is the difference between continuous delivery and continuous deployment?",
    "Statistics":         "Explain the Central Limit Theorem and its importance in data science.",
    "System Design":      "How would you design a highly available distributed cache?",
    "REST API":           "What is idempotency in REST APIs? Which HTTP methods should be idempotent?",
    "Computer Vision":    "What is the role of anchor boxes in object detection models like YOLO?",
}


def get_role_key(role: str) -> str:
    """Map user-provided role to the closest bank key."""
    role_lower = role.lower()
    for key in QUESTION_BANK:
        if key in role_lower or any(w in role_lower for w in key.split()):
            return key
    return None


def generate_questions(
    role: str,
    skills: list,
    experience_level: str,
    total_questions: int = 10,
    api_key: Optional[str] = None,
) -> list:
    """
    Generate a list of interview question dicts.

    Returns:
        [{"id": int, "question": str, "category": str, "difficulty": str}, ...]
    """
    # Try API-based generation first
    if api_key:
        api_qs = _generate_via_api(role, skills, experience_level, total_questions, api_key)
        if api_qs:
            return api_qs

    return _generate_from_bank(role, skills, experience_level, total_questions)


def _generate_via_api(role, skills, experience_level, total_questions, api_key):
    """Attempt Gemini or OpenAI generation. Returns [] on any failure."""
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")

        skills_str = ", ".join(skills[:8]) if skills else role
        prompt = f"""Generate {total_questions} interview questions for a {role} with {experience_level} experience level.
Skills to focus on: {skills_str}

Return ONLY a numbered list in this exact format (no extra text):
1. [Technical] [Medium] What is your question here?
2. [Behavioral] [Easy] What is your question here?

Categories: Technical, Behavioral, Problem Solving, Scenario
Difficulty: Easy, Medium, Hard
Mix difficulties: 30% Easy, 50% Medium, 20% Hard"""

        response = model.generate_content(prompt)
        return _parse_api_response(response.text, total_questions)
    except Exception:
        pass

    try:
        import openai
        client = openai.OpenAI(api_key=api_key)
        skills_str = ", ".join(skills[:8]) if skills else role
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content":
                f"Generate {total_questions} interview questions for {role} ({experience_level}) "
                f"focusing on: {skills_str}. Format each as: [Category] [Difficulty] Question text"}],
            max_tokens=1500,
        )
        return _parse_api_response(resp.choices[0].message.content, total_questions)
    except Exception:
        return []


def _parse_api_response(text: str, n: int) -> list:
    """Parse numbered API response into structured list."""
    import re
    questions = []
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    for i, line in enumerate(lines[:n]):
        # Strip leading number
        line = re.sub(r"^\d+[\.\)]\s*", "", line)
        # Extract category and difficulty if bracketed
        cat_match  = re.search(r"\[(Technical|Behavioral|Problem Solving|Scenario|Project)\]", line, re.I)
        diff_match = re.search(r"\[(Easy|Medium|Hard)\]", line, re.I)
        category   = cat_match.group(1).title()  if cat_match  else "Technical"
        difficulty = diff_match.group(1).title() if diff_match else "Medium"
        # Clean brackets from question text
        question = re.sub(r"\[.*?\]", "", line).strip(" -:")
        if len(question) > 10:
            questions.append({
                "id": i + 1,
                "question":   question,
                "category":   category,
                "difficulty": difficulty,
            })
    return questions if questions else []


def _generate_from_bank(role: str, skills: list, experience_level: str, total: int) -> list:
    """
    Pull questions from the built-in bank.
    Difficulty distribution based on experience level.
    """
    rng = random.Random(hash(role + experience_level))

    dist = {
        "Fresher (0-1 years)":     {"Easy": 0.50, "Medium": 0.40, "Hard": 0.10},
        "Junior (1-3 years)":      {"Easy": 0.30, "Medium": 0.50, "Hard": 0.20},
        "Mid-level (3-6 years)":   {"Easy": 0.15, "Medium": 0.55, "Hard": 0.30},
        "Senior (6-10 years)":     {"Easy": 0.05, "Medium": 0.45, "Hard": 0.50},
        "Principal (10+ years)":   {"Easy": 0.00, "Medium": 0.30, "Hard": 0.70},
    }.get(experience_level, {"Easy": 0.25, "Medium": 0.50, "Hard": 0.25})

    role_key = get_role_key(role)
    pool = {}

    # Merge role-specific + generic
    sources = [GENERIC_QUESTIONS]
    if role_key:
        sources.insert(0, QUESTION_BANK[role_key])

    for source in sources:
        for cat, diffs in source.items():
            if cat not in pool:
                pool[cat] = {"Easy": [], "Medium": [], "Hard": []}
            for diff, qs in diffs.items():
                pool[cat][diff].extend(qs)

    # Flatten with difficulty weighting
    candidates_by_diff = {"Easy": [], "Medium": [], "Hard": []}
    for cat, diffs in pool.items():
        for diff, qs in diffs.items():
            for q in qs:
                candidates_by_diff[diff].append((q, cat, diff))

    # Shuffle each bucket
    for diff in candidates_by_diff:
        rng.shuffle(candidates_by_diff[diff])

    selected = []
    counts = {d: max(1, round(total * p)) for d, p in dist.items()}
    # Adjust to hit exactly total
    while sum(counts.values()) > total:
        counts[max(counts, key=counts.get)] -= 1
    while sum(counts.values()) < total:
        counts[max(dist, key=dist.get)] += 1

    for diff, count in counts.items():
        bucket = candidates_by_diff[diff]
        for item in bucket[:count]:
            selected.append(item)

    # Add skill-specific questions (replace last few)
    skill_qs = []
    for sk in skills:
        if sk in SKILL_QUESTIONS and len(skill_qs) < max(2, total // 5):
            skill_qs.append((SKILL_QUESTIONS[sk], "Technical", "Medium"))

    # Replace tail of selected with skill questions
    if skill_qs:
        selected = selected[:total - len(skill_qs)] + skill_qs

    rng.shuffle(selected)
    selected = selected[:total]

    return [
        {"id": i + 1, "question": q, "category": cat, "difficulty": diff}
        for i, (q, cat, diff) in enumerate(selected)
    ]
