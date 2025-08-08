from langchain.tools import tool
import faiss
import json
import numpy as np
from transformers import CLIPProcessor, CLIPModel
import re
import torch
# Load CLIP model
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def load_caption_index(index_path="captions_faiss.index", metadata_path="captions_metadata.json"):
    index = faiss.read_index(index_path)
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    return index, metadata

def get_clip_embedding_for_text(query: str):
    inputs = clip_processor(text=[query], images=None, return_tensors="pt", padding=True)
    outputs = clip_model.get_text_features(**inputs)
    embedding = outputs[0].detach().numpy()
    embedding = embedding / np.linalg.norm(embedding)
    return embedding.astype("float32")

@tool
def search_caption_with_query(query: str, k: int = 3) -> str:
    """Searches the caption index for images matching the query."""
    index, metadata = load_caption_index()

    # Get page number if mentioned
    match = re.search(r"page\s+(\d+)", query.lower())
    target_page = int(match.group(1)) if match else None

    # Filter metadata by page if needed
    if target_page is not None:
        filtered_metadata = [m for m in metadata if int(m["page_number"]) == target_page]
        if not filtered_metadata:
            return f"❌ No image found for page {target_page}."

        # Create temporary FAISS index for just that page
        temp_index = faiss.IndexFlatL2(index.d)
        temp_vectors = []

        for i, m in enumerate(metadata):
            if int(m["page_number"]) == target_page:
                vector = index.reconstruct(i)
                temp_index.add(np.array([vector], dtype=np.float32))
                temp_vectors.append((vector, m))

        # Embed the query
        inputs = clip_processor(text=[query], return_tensors="pt", padding=True)
        with torch.no_grad():
            embedding = clip_model.get_text_features(**inputs)[0].numpy()
        embedding = embedding / np.linalg.norm(embedding)
        embedding = embedding.astype(np.float32).reshape(1, -1)

        # Search in filtered index
        D, I = temp_index.search(embedding, k)

        results = []
        for idx in I[0]:
            if idx >= len(temp_vectors):
                continue
            meta = temp_vectors[idx][1]
            results.append(
                f"🧠 **Caption**: {meta['caption']}\n"
                f"📄 **Page**: {meta['page_number']}\n"
                f"🖼️ **Image Path**: {meta['image_path']}\n\n---"
            )

        return "\n".join(results) if results else f"❌ No matching image found for page {target_page}."

    # No page mentioned — full index search
    inputs = clip_processor(text=[query], return_tensors="pt", padding=True)
    with torch.no_grad():
        embedding = clip_model.get_text_features(**inputs)[0].numpy()
    embedding = embedding / np.linalg.norm(embedding)
    embedding = embedding.astype(np.float32).reshape(1, -1)

    D, I = index.search(embedding, k)

    results = []
    for idx in I[0]:
        if idx == -1 or idx >= len(metadata):
            continue
        meta = metadata[idx]
        results.append(
            f"🧠 **Caption**: {meta['caption']}\n"
            f"📄 **Page**: {meta['page_number']}\n"
            f"🖼️ **Image Path**: {meta['image_path']}\n\n---"
        )

    return "\n".join(results) if results else "❌ No matching image caption found."
