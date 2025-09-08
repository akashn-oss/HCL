# LLM Resume Reviewer

A Streamlit-based web app that reviews resumes using LLMs (OpenAI GPT-4 by default) and provides tailored, constructive feedback for specific job roles.

---

## 1. Environment Setup

### Create and activate virtual environment (Windows / PyCharm)
```powershell
python -m venv guvi
.\guvi\Scripts\Activate.ps1
```

For Linux/Mac:
```bash
python3 -m venv env
source env/bin/activate
```

---

## 2. Installation

Upgrade pip and install dependencies:
```bash
pip install --upgrade pip
pip install streamlit openai pdfplumber fpdf python-dotenv
```

(Optionally, save dependencies)
```bash
pip freeze > requirements.txt
```

---

## 3. Get OpenAI API Key

1. Go to [OpenAI Platform](https://platform.openai.com/)
2. Sign in or create an account
3. Navigate to [API Keys](https://platform.openai.com/account/api-keys)
4. Click **Create new secret key** and copy it

---

## 4. Provide API Key to App

### `.env` file (recommended)
Create a `.env` file in the same directory as `app.py`:
```
OPENAI_API_KEY=your_api_key_here
```

## 5. Run the App

```bash
streamlit run app.py
```

App will launch at: [http://localhost:8501](http://localhost:8501)

---

## Features

- Upload or paste resumes (PDF/TXT)
- Specify target job role or job description
- LLM feedback: missing skills, clarity improvements, tailored bullets
- Improved resume generation (download as TXT or PDF)
- Privacy: resumes are processed in memory, not stored
