import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from datasets import load_dataset
from PIL import Image
import random
import os

def test_trocr_on_dataset(num_samples=10):
    # 1. Load Model & Processor
    print("Loading TrOCR model...")
    processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
    model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 2. Load Dataset (Test Split)
    print("Loading test samples from Hugging Face...")
    ds = load_dataset("Teklia/IAM-line", split="test")

    # 3. Pick random samples
    indices = random.sample(range(len(ds)), num_samples)
    
    print("\n" + "="*80)
    print(f"{'INDEX':<6} | {'ACTUAL TEXT':<35} | {'TROCR PREDICTION'}")
    print("-" * 80)

    for idx in indices:
        sample = ds[idx]
        actual_text = sample['text']
        
        # Process image
        image = sample['image'].convert("RGB")
        pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)

        # Generate Text
        with torch.no_grad():
            generated_ids = model.generate(pixel_values)
        
        predicted_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
        print(f"{idx:<6} | {actual_text[:35]:<35} | {predicted_text}")
    print("="*80)

if __name__ == "__main__":
    test_trocr_on_dataset(10)
