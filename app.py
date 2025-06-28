import streamlit as st
import requests
import json
import os

# ==============================
# IBM Credentials (Masked)
# ==============================
API_KEY = os.getenv("WATSONX_API_KEY")
PROJECT_ID = "7b0e32e1-7e5b-489c-8508-36736d8feba9"
WATSONX_URL = "https://us-south.ml.cloud.ibm.com"
MODEL_ID = "ibm/granite-3-3-8b-instruct"

def get_iam_token():
    url = "https://iam.cloud.ibm.com/identity/token"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    data = f"apikey={API_KEY}&grant_type=urn:ibm:params:oauth:grant-type:apikey"
    response = requests.post(url, headers=headers, data=data)
    return response.json()["access_token"]

def call_granite(resume, jd):
    token = get_iam_token()
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    prompt = f"""
You are a hiring assistant helping a recruiter evaluate if a candidate is a good fit for a job.

Compare the following:
Resume:
{resume}

Job Description:
{jd}

Return a response in JSON:
{{
  "match_score": number (0–100),
  "summary": "string",
  "missing_skills": ["string"],
  "resume_improvements": ["string"]
}}
"""
    payload = {
        "model_id": MODEL_ID,
        "input": prompt,
        "parameters": {
            "decoding_method": "greedy",
            "max_new_tokens": 300
        },
        "project_id": PROJECT_ID
    }

    url = f"{WATSONX_URL}/ml/v1/text/generation?version=2024-05-14"
    response = requests.post(url, headers=headers, json=payload)

    if response.status_code == 200:
        return response.json()["results"][0]["generated_text"]
    else:
        return f"ERROR: {response.status_code} - {response.text}"


st.set_page_config(page_title="SkillCheck AI", layout="centered")
st.markdown("<h1 style='text-align: center;'>SkillCheck AI</h1>", unsafe_allow_html=True)
st.caption("Powered by IBM Granite • Resume vs JD Match Scorer")

examples = {
    "-- Select Example --": ("", ""),
    "Perfect Fit (Krithi)": (
        "Name: Krithi Raman\nSkills: Python, SQL, Tableau\nExperience: 3.5 years at Deloitte",
        "Looking for a Data Analyst with 3+ years experience, Python, SQL, and dashboarding skills."
    ),
    "Weak Fit (Raj)": (
        "Name: Raj Verma\nSkills: Java, HTML, CSS\nExperience: 2 years in frontend dev",
        "Looking for a Data Scientist with Python, ML, SQL, and 4+ years experience."
    )
}

selected_example = st.selectbox("Load Example", list(examples.keys()))
resume, jd = examples[selected_example]

col1, col2 = st.columns(2)
with col1:
    resume = st.text_area("Resume", resume, height=200)
with col2:
    jd = st.text_area("Job Description", jd, height=200)

if st.button("Evaluate Fit"):
    if not resume.strip() or not jd.strip():
        st.warning("Please enter both resume and job description.")
    else:
        with st.spinner("Calling IBM Granite..."):
            result = call_granite(resume, jd)
        st.code(result)
