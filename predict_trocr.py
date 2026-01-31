import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import argparse
import os

def predict(image_path):
    # 1. Load Image
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return

    image = Image.open(image_path).convert("RGB")
    
    # 2. Load Model & Processor
    print("Loading TrOCR model (microsoft/trocr-base-handwritten)...")
    # This will download the model weights (approx 1.3GB) on the first run
    processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
    model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')

    # 3. Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")

    # 4. Process Image
    print("Transcribing...")
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)

    # 5. Generate Text
    with torch.no_grad():
        generated_ids = model.generate(pixel_values)
    
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    print("\n" + "="*30)
    print(f"IMAGE: {image_path}")
    print(f"RESULT: {generated_text}")
    print("="*30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TrOCR Handwriting Prediction")
    parser.add_argument("-i", "--image", required=True, help="Path to the handwritten image")
    args = parser.parse_args()

    predict(args.image)
