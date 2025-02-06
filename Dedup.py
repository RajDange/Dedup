import pandas as pd
import numpy as np
from rapidfuzz import fuzz
import streamlit as st
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

# Function to calculate fuzzy match score between two rows using rapidfuzz
def fuzzy_match(row1, row2):
    combined1 = ' '.join([str(row1['name']), str(row1['address']), str(row1['city']), str(row1['postal_code'])])
    combined2 = ' '.join([str(row2['name']), str(row2['address']), str(row2['city']), str(row2['postal_code'])])
    return fuzz.ratio(combined1, combined2)

# Function to perform clustering and add cluster info to the DataFrame
def get_clusters(df, threshold=80):
    similarity_scores = []

    for i, row1 in df.iterrows():
        scores = []
        for j, row2 in df.iterrows():
            if i != j:
                score = fuzzy_match(row1, row2)
                if score >= threshold:
                    scores.append((j, score))  # Store index and score if above threshold
        similarity_scores.append(scores)
    
    n = len(df)
    distance_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j, score in similarity_scores[i]:
            distance_matrix[i, j] = 100 - score  # Distance is 100 - similarity score
            distance_matrix[j, i] = distance_matrix[i, j]  # Symmetric
    
    clustering = DBSCAN(metric="precomputed", min_samples=2, eps=20).fit(distance_matrix)
    
    df['cluster_id'] = clustering.labels_
    df['cluster_size'] = df.groupby('cluster_id')['cluster_id'].transform('count')
    df['link_score'] = [max([score for _, score in similarity_scores[i]], default=0) for i in range(n)]
    
    return df

# Streamlit UI components
st.title("Fuzzy Deduplication Tool")

# File upload and delimiter option
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:
    delimiter = st.selectbox("Select CSV Delimiter", [",", ";", "\t","|"], index=0)
    df = pd.read_csv(uploaded_file, delimiter=delimiter)
    
    # Check if necessary columns exist
    required_columns = ['name', 'address', 'city', 'postal_code']
    for col in required_columns:
        if col not in df.columns:
            st.error(f'Missing required column: {col}')
            st.stop()

    # Get fuzzy matching threshold from user
    threshold = st.slider("Set Fuzzy Matching Threshold", 0, 100, 80)

    # Get clusters
    df_with_clusters = get_clusters(df, threshold)

    # Display results in table
    st.subheader("Clustered Data")
    st.write(df_with_clusters[['name', 'address', 'city', 'postal_code', 'cluster_id', 'cluster_size', 'link_score']])

    # Visualize the clusters
    st.subheader("Cluster Size Distribution")
    fig, ax = plt.subplots()
    df_with_clusters['cluster_size'].value_counts().plot(kind='bar', ax=ax)
    ax.set_title("Cluster Size Distribution")
    ax.set_xlabel("Cluster Size")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

    # Download processed file
    csv = df_with_clusters.to_csv(index=False)
    st.download_button(
        label="Download Processed File",
        data=csv,
        file_name="deduplicated_file.csv",
        mime="text/csv"
    )
