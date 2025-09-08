"""
LLM Resume Reviewer - Single-file Streamlit app

Features:
- Upload PDF or paste resume text
- Provide job role and optional job description (paste or upload)
- LLM-powered review using OpenAI (GPT-4 by default)
- Section-wise feedback, missing keywords, suggested improvements
- Option to generate an improved resume
- Download improved resume as TXT or PDF

Requirements (pip):
streamlit
openai
pdfplumber
fpdf
python-dotenv

Set environment variable OPENAI_API_KEY before running, or create a .env file with OPENAI_API_KEY=...

Run:
pip install -r requirements.txt
streamlit run app.py

"""

import os
import io
import json
import textwrap
from typing import Optional

import streamlit as st
import pdfplumber
import openai
from fpdf import FPDF
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4")
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY

def extract_text_from_pdf(file_bytes: bytes) -> str:
    text_parts = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            try:
                page_text = page.extract_text() or ""
            except Exception:
                page_text = ""
            text_parts.append(page_text)
    return "\n\n".join(text_parts)

def clean_text(text: str) -> str:
    return text.strip()

def build_review_prompt(resume_text: str, job_role: str, job_description: Optional[str]) -> str:
    job_role = job_role or "(not specified)"
    jd_section = job_description if job_description else ""
    prompt = f"""
You are an expert career coach and resume writer.  Analyze the provided resume text and the optional job description / target role.
Produce a JSON object ONLY (no extra chatter) with the following fields:

- summary: short (1-3 sentence) assessment of fit for the role.
- scores: object with keys "overall", "formatting", "clarity", "impact", "relevance" each 0-100.
- missing_keywords: list of keywords/skills not present in the resume but commonly expected for the role (short list).
- suggestions: an object with keys for sections (e.g., Experience, Education, Skills, Summary) each containing a short list of concrete suggestions.
- vague_or_redundant: list of examples from the resume that are vague or redundant and how to fix them.
- tailored_bullets: up to 6 suggested improved bullet points rewritten to better match the job role (use realistic quantification and action verbs). If not enough info, make reasonable placeholders like <X%> but keep them natural.
- improved_resume: a plain-text version of the resume with edits applied (concise, formatted for ATS readability). Keep the same structure but improve wording.

Be concise. Use short lists where possible. The JSON should be parseable by a program.

Resume (delimited by triple backticks):
```{resume_text}```

Target job role / job description (delimited):
```Role: {job_role}\n{jd_section}```

Respond now with JSON only.
"""
    max_resume_chars = 35000
    if len(prompt) > max_resume_chars:
        prompt = prompt[:max_resume_chars]
    return prompt

def call_openai_chat(prompt: str, model: str = DEFAULT_MODEL, temperature: float = 0.0) -> str:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set. Set it as environment variable or in .env file.")
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=2500,
        )
        text = response.choices[0].message.content
        return text
    except Exception as e:
        raise

def text_to_pdf_bytes(text: str) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=11)
    for line in text.splitlines():
        wrapped = textwrap.wrap(line, width=95)
        if not wrapped:
            pdf.ln(4)
        for w in wrapped:
            pdf.cell(0, 6, txt=w, ln=1)
    output = io.BytesIO()
    pdf.output(output)
    return output.getvalue()

def main():
    st.set_page_config(page_title="LLM Resume Reviewer", layout="centered")
    st.title("LLM Resume Reviewer — Improve your resume for a target role")
    with st.sidebar:
        st.header("Settings")
        model = st.text_input("OpenAI model", value=DEFAULT_MODEL)
        temperature = st.slider("LLM creativity (temperature)", 0.0, 1.0, 0.0, 0.1)
        show_raw = st.checkbox("Show raw LLM JSON output", value=False)
        privacy = st.checkbox("Don't store uploads on server (recommended)", value=True)
        st.markdown("---")
        st.markdown("**Privacy note:** Uploaded resumes are processed in memory. This app does not persist files to disk by default.")
        st.markdown("Set your OpenAI API key in environment variable OPENAI_API_KEY or create a .env file.")
    st.header("1) Upload or paste your resume")
    uploaded_file = st.file_uploader("Upload PDF or TXT resume", type=["pdf", "txt"] )
    paste_resume = st.text_area("Or paste your resume text here (overrides upload)", height=200)
    st.header("2) Target role / job description")
    job_role = st.text_input("Target job role (e.g., Data Scientist)")
    jd_file = st.file_uploader("Optional: Upload job description (txt or pdf)", type=["pdf","txt"], key="jd")
    jd_paste = st.text_area("Or paste job description here (optional)", height=150)
    if st.button("Run review"):
        with st.spinner("Parsing resume..."):
            resume_text = ""
            if paste_resume and paste_resume.strip():
                resume_text = paste_resume
            elif uploaded_file is not None:
                file_bytes = uploaded_file.read()
                if uploaded_file.type == "application/pdf" or uploaded_file.name.lower().endswith('.pdf'):
                    resume_text = extract_text_from_pdf(file_bytes)
                else:
                    try:
                        resume_text = file_bytes.decode("utf-8")
                    except Exception:
                        resume_text = file_bytes.decode("latin-1")
            else:
                st.error("Please upload or paste a resume.")
                return
            resume_text = clean_text(resume_text)
            job_description_text = ""
            if jd_paste and jd_paste.strip():
                job_description_text = jd_paste
            elif jd_file is not None:
                jd_bytes = jd_file.read()
                if jd_file.type == "application/pdf" or jd_file.name.lower().endswith('.pdf'):
                    job_description_text = extract_text_from_pdf(jd_bytes)
                else:
                    try:
                        job_description_text = jd_bytes.decode("utf-8")
                    except Exception:
                        job_description_text = jd_bytes.decode("latin-1")
        prompt = build_review_prompt(resume_text=resume_text, job_role=job_role, job_description=job_description_text)
        st.info("Running LLM review — this will call OpenAI. Make sure your API key is configured.")
        try:
            llm_output = call_openai_chat(prompt=prompt, model=model, temperature=temperature)
        except Exception as e:
            st.error(f"LLM call failed: {e}")
            return
        parsed = None
        try:
            cleaned = llm_output.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.strip('`')
            parsed = json.loads(cleaned)
        except Exception:
            try:
                start = llm_output.index('{')
                end = llm_output.rindex('}')
                candidate = llm_output[start:end+1]
                parsed = json.loads(candidate)
            except Exception:
                parsed = None
        st.header("Review result")
        if show_raw:
            st.subheader("Raw LLM response")
            st.code(llm_output, language='json')
        if parsed is None:
            st.error("Couldn't parse JSON from the LLM response. Showing raw output below. You can try increasing temperature or changing model.")
            st.subheader("LLM output")
            st.text_area("LLM output", value=llm_output, height=300)
            return
        st.subheader("Summary")
        st.write(parsed.get('summary', '—'))
        st.subheader("Scores")
        scores = parsed.get('scores', {})
        for k, v in scores.items():
            try:
                val = float(v)
            except Exception:
                val = None
            if val is not None:
                st.write(f"**{k.capitalize()}:** {val:.0f}/100")
                st.progress(int(max(0, min(100, val))))
            else:
                st.write(f"**{k.capitalize()}:** {v}")
        st.subheader("Missing keywords / skills")
        missing = parsed.get('missing_keywords', [])
        if missing:
            st.write(', '.join(missing))
        else:
            st.write("None detected or not provided.")
        st.subheader("Suggestions by section")
        suggestions = parsed.get('suggestions', {})
        for section, tips in suggestions.items():
            st.markdown(f"**{section}**")
            if isinstance(tips, list):
                for t in tips:
                    st.write(f"- {t}")
            else:
                st.write(tips)
        st.subheader("Vague or redundant items")
        vag = parsed.get('vague_or_redundant', [])
        if vag:
            for item in vag:
                st.write(f"- {item}")
        else:
            st.write("None found.")
        st.subheader("Tailored bullets (up to 6)")
        bullets = parsed.get('tailored_bullets', [])
        if bullets:
            for b in bullets:
                st.write(f"- {b}")
        st.subheader("Improved resume (plain text)")
        improved = parsed.get('improved_resume', '')
        st.text_area("Improved resume (editable)", value=improved, height=400, key="improved")
        st.markdown("---")
        st.subheader("Download improved resume")
        improved_text = improved or ""
        st.download_button("Download TXT", data=improved_text.encode('utf-8'), file_name='improved_resume.txt')
        try:
            pdf_bytes = text_to_pdf_bytes(improved_text)
            st.download_button("Download PDF", data=pdf_bytes, file_name='improved_resume.pdf')
        except Exception:
            st.write("PDF generation failed on this platform.")
        st.success("Done — review the suggestions and edit the improved resume as needed.")

if __name__ == '__main__':
    main()
