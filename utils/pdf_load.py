import fitz  # PyMuPDF
import os
import logging

def extract_images(pdf_path, output_dir):
    print(f"📄 Extracting images from {pdf_path} to {output_dir}...")
    doc = fitz.open(pdf_path)
    os.makedirs(output_dir, exist_ok=True)
    
    image_count = 0
    for page_num in range(len(doc)):
        page = doc[page_num]
        images = page.get_images(full=True)
        
        for img_index, img in enumerate(images):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            
            filename = f"{page_num+1}_img{img_index+1}.{image_ext}"
            with open(os.path.join(output_dir, filename), "wb") as f:
                f.write(image_bytes)
                
            image_count += 1
    
    print(f"✅ Extracted {image_count} images.")
    logging.info(f"Extracted {image_count} images from {pdf_path} to {output_dir}.")