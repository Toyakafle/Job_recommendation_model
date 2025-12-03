import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib


# Load jobs dataset
jobs = pd.read_csv('random_1000_jobs.csv')


# Fit TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(jobs['job_description'])


# Save vectorizer and matrix
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
joblib.dump(tfidf_matrix, 'jobs_tfidf_matrix.pkl')
print("TF-IDF model and job matrix saved.")
