import os
import uuid
import json
import faiss
import numpy as np
from PIL import Image
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from transformers import CLIPProcessor, CLIPModel
import logging
import re

# Load models
ocr_model = ocr_predictor(pretrained=True)
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Constants
dimension = 512
metadata_store = []
faiss_index = faiss.IndexFlatL2(dimension)

def extract_ocr_text(image_path):
    doc = DocumentFile.from_images(image_path)
    result = ocr_model(doc).export()
    blocks = result['pages'][0]['blocks']
    caption = " ".join(
        word['value']
        for block in blocks if block.get("lines")
        for line in block["lines"] if line.get("words")
        for word in line["words"]
    ) if blocks else "No OCR text detected"
    return caption.strip()

def generate_clip_embedding(image_path, caption):
    image = Image.open(image_path).convert("RGB")
    inputs = clip_processor(text=[caption], images=image, return_tensors="pt", padding=True)
    outputs = clip_model(**inputs)
    embedding = outputs.text_embeds[0].detach().numpy()
    embedding = embedding / np.linalg.norm(embedding)
    return embedding

def add_to_index(caption, embedding, page_number, image_path):
    doc_id = str(uuid.uuid4())
    vector = np.array([embedding], dtype=np.float32)
    faiss_index.add(vector)
    metadata_store.append({
        "id": doc_id,
        "caption": caption,
        "page_number": page_number,
        "image_path": image_path
    })
    return doc_id

def save_index(index_path="captions_faiss.index", metadata_path="captions_metadata.json"):
    faiss.write_index(faiss_index, index_path)
    with open(metadata_path, "w") as f:
        json.dump(metadata_store, f)
    print("💾 Saved index + metadata.")

def process_images_and_build_index(image_dir):
    for file in os.listdir(image_dir):
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            image_path = os.path.join(image_dir, file)
            print(f"📄 Processing: {file}")
            caption = extract_ocr_text(image_path)
            embedding = generate_clip_embedding(image_path, caption)
            
            match = re.search(r'(\d+)_img\d+', file)
            page_number = int(match.group(1)) if match else -1
            doc_id = add_to_index(caption, embedding, page_number, image_path)
            print(f"✅ Indexed: {doc_id} | Caption: {caption[:50]}...")
        
        logging.info(f"Processed {file} with caption: {caption[:50]}...")
    save_index()


