import torch
print("GPU available:", torch.cuda.is_available())

from sentence_transformers import SentenceTransformer, util
import json
from tqdm import tqdm
import os

# Load the high-performance semantic model
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
model = SentenceTransformer("sentence-transformers/gtr-t5-large", device=device)

# Seed sentences to define the concept of interest
seed_sentences = [
    "Flüchtlinge brauchen Hilfe.",
    "Migration nach Deutschland nimmt zu.",
    "Asylbewerber suchen Schutz.",
    "Die Einwanderungspolitik wird diskutiert.",
    "Geflüchtete Menschen aus Syrien.",
    "Asylanträge steigen.",
    "Der Umgang mit Migranten in Europa.",
    "Integration von Flüchtlingen.",
    "Zuwanderung und Aufnahmezentren in Deutschland.",
]

# Encode once
seed_embeddings = model.encode(seed_sentences, convert_to_tensor=True, normalize_embeddings=True)

# Threshold for semantic similarity
SIMILARITY_THRESHOLD = 0.75

# Optional prefilter with expanded stemmed keywords
import re
keyword_pattern = re.compile(
    r'\b(migr|flücht|asyl|einwander|zuwander|ausländer|geflücht|schutzsuch|aufnahmezentrum)\w*', re.IGNORECASE)

def is_semantically_relevant(sentences_batch):
    embeddings = model.encode(sentences_batch, convert_to_tensor=True, normalize_embeddings=True)
    scores = util.cos_sim(embeddings, seed_embeddings)
    max_scores = scores.max(dim=1).values
    return max_scores >= SIMILARITY_THRESHOLD

def filter_file(input_path, output_path, batch_size=128):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    relevant = []
    buffer = []
    meta = []

    for record in tqdm(data):
        sentence = record["sentence"]
        if keyword_pattern.search(sentence):
            buffer.append(sentence)
            print(sentence)
            meta.append(record)
            if len(buffer) >= batch_size:
                mask = is_semantically_relevant(buffer)
                relevant.extend([meta[i] for i in range(len(mask)) if mask[i]])
                buffer, meta = [], []

    # Final batch
    if buffer:
        mask = is_semantically_relevant(buffer)
        relevant.extend([meta[i] for i in range(len(mask)) if mask[i]])

    with open(output_path, "w", encoding="utf-8") as out:
        json.dump(relevant, out, ensure_ascii=False, indent=2)


years = list(range(2010, 2025)) # Generates years from 2010 to 2024

# Base directory for input and output files
base_input_dir = "/content/drive/MyDrive/deu_new_Leipzig_structured/structured_data/"
base_output_dir = "/content/drive/MyDrive/deu_new_Leipzig_structured/filtered_data/"

# Loop through the years and process each file
for year in years:
    input_file = f"{base_input_dir}deu_{year}.json"
    output_file = f"{base_output_dir}deu_{year}.json"

    # Call the filter_file function for each year
    filter_file(input_file, output_file)

print("Filtering process completed for all years.")