# Auction Assistant — Streamlit demo (offline retrieval)

This repository is a demo Streamlit app that:
- Loads your CSV datasets from local disk (configured paths)
- Auto-generates Q/A pairs from rows
- Builds a TF-IDF retrieval model over generated questions
- Lets you ask natural-language queries and returns best-match answers & images (local files)

## Expected local files (these are the exact paths used in the demo)
- `/mnt/data/car.csv`
- `/mnt/data/furniture.csv`
- `/mnt/data/electronics.csv`
- `/mnt/data/antique.csv`
- `/mnt/data/car_images/` (images named `1.jpg` .. `40.jpg`)

> If your files are in different locations, edit `CSV_PATHS` and `IMAGES_DIR` in `app.py`.

## Run locally
1. Create a Python environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
