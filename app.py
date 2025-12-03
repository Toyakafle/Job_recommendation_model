import streamlit as st
import pandas as pd
import PyPDF2
import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(page_title="Hybrid Job Recommendation", layout="wide")
st.title("Hybrid Job Recommendation System")


# ----------------------------
# Load Data and Models
# ----------------------------
jobs = pd.read_csv("random_1000_jobs.csv")
interactions = pd.read_csv("random_user_job_interactions.csv")

vectorizer = joblib.load('tfidf_vectorizer.pkl')
tfidf_matrix = joblib.load('jobs_tfidf_matrix.pkl')


# Display All Jobs at the Top
# ----------------------------
st.subheader("All Available Jobs")
# Show job_title, job_description (or any columns you want)
st.dataframe(jobs[['job_title','job_description']])

st.subheader("Upload your resume (PDF) to get personalized job recommendations.")

# ----------------------------
# Functions
# ----------------------------
def extract_text_from_pdf(uploaded_file):
    reader = PyPDF2.PdfReader(uploaded_file)
    text = "".join([page.extract_text() for page in reader.pages])
    return text

def find_skill_gaps(resume_text, job_text):
    resume_words = set(resume_text.lower().split())
    job_words = set(job_text.lower().split())
    return list(job_words - resume_words)

def collaborative_score(user_job_df, num_jobs):
    job_counts = user_job_df['job_id'].value_counts().to_dict()
    scores = [job_counts.get(job_id, 0) / max(list(job_counts.values()) + [1]) for job_id in range(1, num_jobs+1)]
    return scores

# ----------------------------
# File Upload
# ----------------------------
resume_text = None  # initialize variable

uploaded_file = st.file_uploader("Upload your resume", type=["pdf"])
if uploaded_file:
    resume_text = extract_text_from_pdf(uploaded_file)
    st.markdown(f"### Extracted Resume Text: {resume_text[0:]}")
    st.success("Resume uploaded successfully!")

# ----------------------------
# Recommendations
# ----------------------------

if resume_text:
    # Content-based recommendation
    resume_vec = vectorizer.transform([resume_text])
    content_scores = cosine_similarity(resume_vec, tfidf_matrix)[0]

    # Collaborative recommendation
    collab_scores = collaborative_score(interactions, num_jobs=len(jobs))
    collab_scores = np.array(collab_scores)  # convert list to numpy array

    # Hybrid scoring
    alpha = 0.7
    hybrid_scores = alpha * content_scores + (1-alpha) * collab_scores
    jobs['match_score'] = hybrid_scores

if st.button("Get Job Recommendations"):
    # Top 3 jobs
    top_3_jobs = jobs.sort_values(by='match_score', ascending=False).head(3)
    st.subheader("Top 3 Recommended Jobs")
    for _, row in top_3_jobs.iterrows():
        st.markdown(f"### {row['job_title']}")
        st.progress(min(float(row['match_score']), 1.0))
        st.write(f"Match Score: {round(row['match_score']*100, 2)}%")

    # Skill gaps for top job
    top_job = top_3_jobs.iloc[0]
    gaps = find_skill_gaps(resume_text, top_job['job_description'])
    st.subheader("Skill Gaps for Top Job")
    st.write(gaps if gaps else "No major skill gaps detected")
