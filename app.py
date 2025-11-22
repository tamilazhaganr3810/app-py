# app.py
import streamlit as st
import pandas as pd
import numpy as np
import os
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

st.set_page_config(page_title="Auction Assistant (Streamlit demo)", layout="wide")

# --------- CONFIG: local dataset paths (these are the exact local paths you uploaded) ----------
CSV_PATHS = [
    "/mnt/data/car.csv",
    "/mnt/data/furniture.csv",
    "/mnt/data/electronics.csv",
    "/mnt/data/antique.csv"
]
IMAGES_DIR = "/mnt/data/car_images"  # folder where your 1.jpg..40.jpg were extracted

# ---------- Helper: load CSVs that exist ----------
def load_csvs(paths):
    dfs = []
    for p in paths:
        if os.path.exists(p):
            try:
                df = pd.read_csv(p)
                df['__source_file'] = os.path.basename(p)
                dfs.append(df)
            except Exception as e:
                st.warning(f"Could not read {p}: {e}")
        else:
            st.info(f"CSV not found (skipped): {p}")
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    else:
        return pd.DataFrame()

# ---------- Helper: produce text description + generate question templates ----------
def make_qa_pairs_for_row(row):
    pairs = []
    name = str(row.get('Name','')).strip()
    price = row.get('Price','')
    date = row.get('Date','')
    time_val = row.get('Time','')
    location = row.get('Location','')
    category = row.get('Category','')
    year = row.get('Year','')
    # create structured answer
    answer_lines = []
    if name:
        answer_lines.append(f"Name: {name}")
    if category:
        answer_lines.append(f"Category: {category}")
    if year and not (pd.isna(year) or str(year).strip()==""):
        answer_lines.append(f"Year: {year}")
    if price != '' and not pd.isna(price):
        answer_lines.append(f"Price: {price}")
    if date and not pd.isna(date):
        answer_lines.append(f"Date: {date}")
    if time_val and not pd.isna(time_val):
        answer_lines.append(f"Time: {time_val}")
    if location and not pd.isna(location):
        answer_lines.append(f"Location: {location}")
    img = row.get('ImageURL') or row.get('Image') or row.get('image') or ''
    if isinstance(img, str) and img.strip():
        # show image filename so app can resolve the path
        answer_lines.append(f"Image: {img.strip()}")
    answer_full = "\n".join(answer_lines).strip()
    if not answer_full:
        # fallback: create an answer from all fields
        answer_full = " | ".join([f"{k}: {v}" for k,v in row.items() if k != '__source_file'])
    # question templates
    templates = []
    if name:
        templates = [
            f"Show me details about {name}",
            f"Tell me about {name}",
            f"What is the price of {name}?",
            f"Where is the auction for {name}?",
            f"When is the auction for {name}?",
            f"Give me the info for {name}",
            f"Details of {name}",
            f"{name} details",
            f"How much is {name}?"
        ]
    else:
        templates = ["Show me auctions", "List auctions", "What auctions are available?"]
    for q in templates:
        pairs.append((q, answer_full))
    return pairs

# ---------- Build QA bank ----------
@st.cache_data(show_spinner=False)
def build_qa_bank(df):
    qa_pairs = []
    for _, row in df.iterrows():
        qa_pairs.extend(make_qa_pairs_for_row(row))
    qa_df = pd.DataFrame(qa_pairs, columns=['question','answer'])
    return qa_df

# ---------- Build TF-IDF index ----------
@st.cache_resource(show_spinner=False)
def build_tfidf_index(questions):
    vect = TfidfVectorizer(ngram_range=(1,2), max_features=5000)
    q_matrix = vect.fit_transform(questions)
    return vect, q_matrix

# ---------- Main UI ----------
st.title("🧠 Auction Assistant — Streamlit demo (offline retrieval)")

with st.sidebar:
    st.header("Data sources")
    for p in CSV_PATHS:
        exists = os.path.exists(p)
        st.write(f"- `{p}` — {'FOUND' if exists else 'MISSING'}")
    st.write("---")
    st.write("Images dir:")
    st.write(f"- `{IMAGES_DIR}` — {'FOUND' if os.path.exists(IMAGES_DIR) else 'MISSING'}")
    st.write("---")
    st.caption("This demo uses TF-IDF retrieval — no external internet calls. Use local files as shown above.")

# load data
df = load_csvs(CSV_PATHS)
if df.empty:
    st.error("No CSVs were found in the configured locations. Place your CSVs and re-run.")
    st.stop()

st.success(f"Loaded combined dataset — rows: {len(df)}")

# show basic preview
st.subheader("Dataset preview")
with st.expander("View raw data"):
    st.dataframe(df.head(20))

# Build QA bank and index
qa_df = build_qa_bank(df)
st.info(f"Generated {len(qa_df)} QA pairs from the dataset.")
if st.checkbox("Preview QA pairs (sample)", value=False):
    st.dataframe(qa_df.sample(min(10, len(qa_df))))

# Build TF-IDF
vectorizer, q_matrix = build_tfidf_index(qa_df['question'].tolist())

# Query UI
st.subheader("Ask the assistant")
question = st.text_input("Enter your question (e.g. 'What is the price of Hyundai i20?')", "")
top_k = st.slider("Top K results", min_value=1, max_value=5, value=1)

if st.button("Get answer") and question.strip():
    # vectorize
    q_vec = vectorizer.transform([question])
    # cosine similarity via linear_kernel
    cosine_similarities = linear_kernel(q_vec, q_matrix).flatten()
    best_idxs = np.argsort(-cosine_similarities)[:top_k]
    results = []
    for idx in best_idxs:
        score = float(cosine_similarities[idx])
        question_match = qa_df.iloc[idx]['question']
        answer = qa_df.iloc[idx]['answer']
        results.append((score, question_match, answer))
    # show results
    for i, (score, qmatch, answer) in enumerate(results, start=1):
        st.markdown(f"**Result #{i} — score:** `{score:.4f}`")
        st.markdown(f"- **Matched template:** {qmatch}")
        st.markdown(f"- **Answer:**")
        st.code(answer, language="text")
        # try to find image filename inside answer
        # look for "Image: <name>"
        lines = answer.splitlines()
        img_name = None
        for L in lines:
            if L.startswith("Image:"):
                img_name = L.split("Image:",1)[1].strip()
                break
        if img_name:
            # resolve path by checking IMAGES_DIR and CSV filenames
            possible_paths = [
                os.path.join(IMAGES_DIR, img_name),
                os.path.join("/mnt/data", img_name),
                img_name  # maybe already a path
            ]
            found = False
            for p in possible_paths:
                if os.path.exists(p):
                    st.image(p, caption=f"Image: {os.path.basename(p)}", use_column_width=True)
                    found = True
                    break
            if not found:
                st.warning(f"Image not found in expected places. Expected: {possible_paths}")
        st.write("---")

# Gallery
st.subheader("Image gallery (local images)")
if os.path.exists(IMAGES_DIR):
    image_files = sorted([f for f in os.listdir(IMAGES_DIR) if f.lower().endswith(('.png','.jpg','.jpeg'))])
    if image_files:
        cols = st.columns(4)
        for i, fname in enumerate(image_files):
            path = os.path.join(IMAGES_DIR, fname)
            col = cols[i % 4]
            try:
                col.image(path, caption=fname)
            except Exception as e:
                col.write(fname)
                col.write("Could not display image.")
    else:
        st.info("No images found in the directory.")
else:
    st.info("Images directory not found.")

st.markdown("---")
st.caption("Repo simulation: load CSVs from local `/mnt/data/` and images from `/mnt/data/car_images`. To run: `streamlit run app.py`")
