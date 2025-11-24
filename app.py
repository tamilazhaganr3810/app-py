import streamlit as st
import pandas as pd
import numpy as np
import os
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

st.set_page_config(
    page_title="Auction Assistant — Streamlit Demo",
    layout="wide"
)

# ------------------------------
# CONFIG: Use repo-relative paths
# ------------------------------

CSV_PATHS = [
    "car.csv",
    "furniture.csv",
    "electronics.csv",
    "antique.csv"
]

# Folder that contains images
IMAGES_DIR = "car_images"      # <--- IMPORTANT

# ------------------------------
# Load CSV files
# ------------------------------
def load_csvs(paths):
    dfs = []
    for p in paths:
        if os.path.exists(p):
            df = pd.read_csv(p)
            df["__source_file"] = os.path.basename(p)
            dfs.append(df)
        else:
            st.warning(f"CSV not found: {p}")
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame()


# ------------------------------
# Generate Q/A pairs
# ------------------------------
def make_qa_pairs(row):
    pairs = []

    name = str(row.get("Name", "")).strip()
    price = row.get("Price", "")
    date = row.get("Date", "")
    time_val = row.get("Time", "")
    location = row.get("Location", "")
    category = row.get("Category", "")
    year = row.get("Year", "")
    image = str(row.get("ImageURL", "")).strip()

    # Build answer
    answer = ""

    if name: answer += f"Name: {name}\n"
    if category: answer += f"Category: {category}\n"
    if year: answer += f"Year: {year}\n"
    if price: answer += f"Price: {price}\n"
    if date: answer += f"Date: {date}\n"
    if time_val: answer += f"Time: {time_val}\n"
    if location: answer += f"Location: {location}\n"
    if image: answer += f"Image: {image}\n"

    answer = answer.strip()

    # Create templates
    if name:
        templates = [
            f"Show me details about {name}",
            f"Tell me about {name}",
            f"What is the price of {name}?",
            f"When is the auction for {name}?",
            f"Where is the auction for {name}?",
            f"Details of {name}"
        ]
    else:
        templates = ["Show auctions", "List auctions"]

    for q in templates:
        pairs.append((q, answer))

    return pairs


def build_qa_bank(df):
    qa = []
    for _, row in df.iterrows():
        qa.extend(make_qa_pairs(row))
    return pd.DataFrame(qa, columns=["question", "answer"])


# ------------------------------
# TF-IDF index
# ------------------------------
def build_tfidf(questions):
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
    matrix = vectorizer.fit_transform(questions)
    return vectorizer, matrix


# ------------------------------
# STREAMLIT UI
# ------------------------------
st.sidebar.header("Data sources")

for p in CSV_PATHS:
    st.sidebar.write(f"- {p} — {'FOUND' if os.path.exists(p) else 'MISSING'}")

st.sidebar.write("---")

# Images folder status
if os.path.exists(IMAGES_DIR):
    st.sidebar.success(f"{IMAGES_DIR} — FOUND")
else:
    st.sidebar.error(f"{IMAGES_DIR} — MISSING (create folder and add images)")

st.title("🧠 Auction Assistant — Streamlit demo (offline retrieval)")

# Load datasets
df = load_csvs(CSV_PATHS)

if df.empty:
    st.error("No CSVs found! Add your CSV files to the repository and reload.")
    st.stop()

st.success(f"Loaded dataset: {len(df)} rows")

# Show raw data
with st.expander("View raw data"):
    st.dataframe(df)

# Build QA dataset
qa_df = build_qa_bank(df)
st.info(f"Generated {len(qa_df)} training Q/A pairs.")

# TF–IDF
vectorizer, q_matrix = build_tfidf(qa_df["question"].tolist())

# ------------------------------
# USER QUERY
# ------------------------------
st.subheader("Ask the AI assistant")

query = st.text_input("Type your question")

if st.button("Get answer"):
    if not query.strip():
        st.warning("Enter a question.")
    else:
        q_vec = vectorizer.transform([query])
        similarity = linear_kernel(q_vec, q_matrix).flatten()

        top_index = similarity.argmax()
        best_score = similarity[top_index]

        st.write(f"**Matched question:** {qa_df.iloc[top_index]['question']}")
        st.write(f"**Confidence score:** `{best_score:.4f}`")

        answer = qa_df.iloc[top_index]["answer"]
        st.code(answer)

        # ------------------------------
        # IMAGE HANDLING
        # ------------------------------
        lines = answer.split("\n")
        image_file = None

        for line in lines:
            if line.startswith("Image:"):
                image_file = line.replace("Image:", "").strip()
                break

        if image_file:
            img_path = os.path.join(IMAGES_DIR, image_file)

            if os.path.exists(img_path):
                st.image(img_path, caption=image_file, use_container_width=True)
            else:
                st.warning(f"Image file not found: {img_path}")

# ------------------------------
# IMAGE GALLERY (OPTIONAL)
# ------------------------------
st.subheader("Available images (Gallery)")

if os.path.exists(IMAGES_DIR):
    img_files = sorted(os.listdir(IMAGES_DIR))
    cols = st.columns(4)

    for i, img in enumerate(img_files):
        if img.lower().endswith((".jpg", ".png", ".jpeg")):
            cols[i % 4].image(os.path.join(IMAGES_DIR, img), caption=img)
else:
    st.info("Place your images inside a folder named 'car_images'")
