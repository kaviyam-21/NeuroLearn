import torch
import numpy as np
from datasets import load_dataset
from models.pytorch_models import CNNBiLSTM
from utilities.iam_dataset import resize_image
from predict import decode
import random

# Configuration
ALPHABET = r' !"#&\'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "best_model.pth"

def test_samples(num_samples=5):
    # 1. Load Model
    net = CNNBiLSTM(alphabet_size=len(ALPHABET) + 1).to(DEVICE)
    try:
        net.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        net.eval()
        print(f"Model loaded from {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 2. Load Dataset (Test Split)
    print("Loading test samples from Hugging Face...")
    ds = load_dataset("Teklia/IAM-line", split="test")

    # 3. Pick random samples
    indices = random.sample(range(len(ds)), num_samples)
    
    print("\n" + "="*50)
    print(f"{'INDEX':<6} | {'ACTUAL TEXT':<30} | {'PREDICTED TEXT'}")
    print("="*50)

    for idx in indices:
        sample = ds[idx]
        actual_text = sample['text']
        
        # Preprocess image
        image = sample['image'].convert('L')
        image = np.array(image)
        
        # Resize/Normalize as in training
        image, _ = resize_image(image, (60, 800))
        image = image.astype(np.float32) / 255.
        image = (image - 0.9425) / 0.1593
        
        # Tensorize
        image_tensor = torch.tensor(image).unsqueeze(0).unsqueeze(0).to(DEVICE)
        
        # Predict
        with torch.no_grad():
            output = net(image_tensor)
            probs = torch.softmax(output, dim=-1).cpu().numpy()
            predicted_text = decode(probs)[0]
            
        print(f"{idx:<6} | {actual_text[:30]:<30} | {predicted_text}")

if __name__ == "__main__":
    test_samples(10)
