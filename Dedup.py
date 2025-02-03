import pandas as pd
import numpy as np
import multiprocessing as mp
from rapidfuzz import fuzz
from metaphone import doublemetaphone
import textdistance
from tqdm import tqdm
import streamlit as st
from typing import List, Dict

# Address normalization dictionary
ADDRESS_MAP = {
    "street": "st",
    "avenue": "ave",
    "boulevard": "blvd",
    "road": "rd",
    "lane": "ln",
    "drive": "dr",
    "court": "ct",
    "square": "sq"
}

# Configurable matching thresholds (adjust per field)
THRESHOLDS = {
    "name": 85,
    "address": 90,
    "city": 90,
    "postal_code": 95
}

# Selectable matching techniques
MATCHING_TECHNIQUES = {
    "levenshtein": textdistance.levenshtein.normalized_similarity,
    "jaro_winkler": textdistance.jaro_winkler.normalized_similarity,
    "cosine": textdistance.cosine.normalized_similarity
}

def normalize_text(text: str) -> str:
    """ Normalize text: lower case, remove extra spaces, and standardize addresses. """
    text = str(text).lower().strip()
    for word, abbrev in ADDRESS_MAP.items():
        text = text.replace(word, abbrev)
    return text

def phonetic_encode(name: str) -> str:
    """ Convert name to phonetic code (Metaphone). """
    return doublemetaphone(name)[0]  # Primary Metaphone encoding

def compute_similarity(val1: str, val2: str, method: str = "levenshtein") -> float:
    """ Compute similarity score using the selected matching technique. """
    if method in MATCHING_TECHNIQUES:
        return MATCHING_TECHNIQUES[method](val1, val2) * 100  # Convert to percentage
    return fuzz.ratio(val1, val2)  # Default to RapidFuzz

def is_similar(record1: Dict[str, str], record2: Dict[str, str]) -> bool:
    """ Check if two records are duplicates using field-specific thresholds. """
    name_score = compute_similarity(record1['name_phonetic'], record2['name_phonetic'], "jaro_winkler")
    address_score = compute_similarity(record1['address'], record2['address'], "levenshtein")
    city_score = compute_similarity(record1['city'], record2['city'], "levenshtein")
    postal_score = compute_similarity(record1['postal_code'], record2['postal_code'], "levenshtein")

    return (name_score >= THRESHOLDS["name"] and
            address_score >= THRESHOLDS["address"] and
            city_score >= THRESHOLDS["city"] and
            postal_score >= THRESHOLDS["postal_code"])

def assign_clusters(records: List[Dict[str, str]]) -> List[int]:
    """ Assigns cluster IDs to similar records. """
    cluster_id = 1
    clusters = [-1] * len(records)

    for i in tqdm(range(len(records)), desc="Clustering"):
        if clusters[i] != -1:
            continue  # Skip already assigned

        clusters[i] = cluster_id
        cluster_size = 1

        for j in range(i + 1, len(records)):
            if clusters[j] == -1 and is_similar(records[i], records[j]):
                clusters[j] = cluster_id
                cluster_size += 1

        # Assign cluster size to all records in the cluster
        for k in range(len(clusters)):
            if clusters[k] == cluster_id:
                records[k]['cluster_size'] = cluster_size

        cluster_id += 1  # Move to the next cluster

    return clusters

def process_chunk(df_chunk: pd.DataFrame) -> pd.DataFrame:
    """ Process a chunk of data in parallel. """
    df_chunk['name_phonetic'] = df_chunk['name'].apply(phonetic_encode)
    df_chunk['address'] = df_chunk['address'].apply(normalize_text)
    df_chunk['city'] = df_chunk['city'].str.lower()
    df_chunk['postal_code'] = df_chunk['postal_code'].str.lower()

    records = df_chunk.to_dict('records')
    df_chunk['cluster_id'] = assign_clusters(records)

    return df_chunk

def run_deduplication(input_file: str, threshold_name: int, threshold_address: int, threshold_city: int, threshold_postal: int, matching_technique: str, delimiter: str) -> pd.DataFrame:
    """ Main function to handle large datasets efficiently. """

    df = pd.read_csv(input_file, delimiter=delimiter).fillna("")
    
    # Update thresholds dynamically
    global THRESHOLDS
    THRESHOLDS = {
        "name": threshold_name,
        "address": threshold_address,
        "city": threshold_city,
        "postal_code": threshold_postal
    }

    # Ensure df is not empty before proceeding
    if df.empty:
        st.error("No valid data available for processing.")
        return df

    num_chunks = mp.cpu_count()  # Max CPU utilization
    chunk_size = max(1, len(df) // num_chunks)
    chunks = [df.iloc[i:i + chunk_size] for i in range(0, len(df), chunk_size)]

    with mp.Pool(num_chunks) as pool:
        results = pool.map(process_chunk, chunks)

    final_df = pd.concat(results).sort_index()
    return final_df

def main():
    st.title('Deduplication Tool')

    # File Upload
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    # Delimiter input
    delimiter = st.selectbox("Select Delimiter", [",", ";", "\t", "|"], index=0)

    if uploaded_file is not None:
        try:
            # Read uploaded file to show sample, with UTF-8 encoding
            df = pd.read_csv(uploaded_file, delimiter=delimiter, encoding='utf-8')
            st.write("Sample of Uploaded Data", df.head())
        except Exception as e:
            st.error(f"Error reading the file: {e}")
            return

        # Matching technique selection
        matching_technique = st.selectbox(
            "Choose Matching Technique",
            ["levenshtein", "jaro_winkler", "cosine"]
        )

        # Threshold inputs for each field
        threshold_name = st.slider("Threshold for Name Similarity", 0, 100, 85)
        threshold_address = st.slider("Threshold for Address Similarity", 0, 100, 90)
        threshold_city = st.slider("Threshold for City Similarity", 0, 100, 90)
        threshold_postal = st.slider("Threshold for Postal Code Similarity", 0, 100, 95)

        # Run deduplication
        if st.button("Run Deduplication"):
            st.write("Running deduplication...")
            final_df = run_deduplication(
                uploaded_file,
                threshold_name,
                threshold_address,
                threshold_city,
                threshold_postal,
                matching_technique,
                delimiter
            )

            # Display output and download option
            if not final_df.empty:
                st.write("Deduplicated Data", final_df)

                # Option to download CSV file
                st.download_button(
                    label="Download Deduplicated CSV",
                    data=final_df.to_csv(index=False, sep=delimiter),
                    file_name="deduplicated_output.csv",
                    mime="text/csv"
                )

if __name__ == "__main__":
    main()