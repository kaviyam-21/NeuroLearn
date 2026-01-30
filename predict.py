import torch
import cv2
import numpy as np
import argparse
import os
from models.pytorch_models import CNNBiLSTM
from utilities.iam_dataset import resize_image

# Alphabet used in training
alphabet_encoding = r' !"#&\'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
alphabet_dict = {alphabet_encoding[i]:i for i in range(len(alphabet_encoding))}

def decode(prediction):
    results = []
    # prediction shape: (Seq_len, 1, Alphabet_size) or (1, Seq_len, Alphabet_size)
    # We expect (N, Seq_len, Alphabet_size) from our forward pass
    for word in prediction:
        result = []
        # Get the predicted character indices (greedy)
        word_indices = np.argmax(word, axis=-1)
        for i, index in enumerate(word_indices):
            # CTC Blank: len(alphabet_encoding)
            if i < len(word_indices) - 1 and word_indices[i] == word_indices[i+1] and word_indices[i] != len(alphabet_encoding):
                continue
            if index == len(alphabet_encoding):
                continue
            else:
                result.append(alphabet_encoding[int(index)])
        results.append(''.join(result))
    return results

def preprocess(image_path, line_or_word):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not find image at {image_path}")
    
    # Target sizes used in training
    target_size = (60, 800) if line_or_word == "line" else (30, 400)
    image, _ = resize_image(image, target_size)
    
    # Normalization
    image = image.astype(np.float32) / 255.
    image = (image - 0.9425) / 0.1593
    
    # Add Batch and Channel dimensions: (1, 1, H, W)
    image = torch.tensor(image).unsqueeze(0).unsqueeze(0)
    return image

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Handwriting Recognition Inference Script")
    parser.add_argument("-i", "--image", required=True, help="Path to the handwriting image file")
    parser.add_argument("-m", "--model", default="best_model.pth", help="Path to the saved model weights")
    parser.add_argument("-t", "--type", choices=["line", "word"], default="line", help="Type of recognition (line or word)")
    args = parser.parse_args()

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize model
    # Alphabet size + 1 for CTC blank
    model = CNNBiLSTM(alphabet_size=len(alphabet_encoding) + 1).to(device)
    
    # Load weights
    if not os.path.exists(args.model):
        print(f"ERROR: Model weights not found at {args.model}. Please finish training first.")
    else:
        model.load_state_dict(torch.load(args.model, map_location=device))
        model.eval()
        print(f"Model loaded from {args.model}")

        # Preprocess
        try:
            image_tensor = preprocess(args.image, args.type).to(device)
            
            # Predict
            with torch.no_grad():
                output = model(image_tensor)
                # output shape: (1, Seq_len, Alphabet_size)
                # We apply softmax to get probabilities (optional for greedy decode but good practice)
                probs = torch.softmax(output, dim=-1).cpu().numpy()
                
            # Decode
            transcription = decode(probs)
            print("-" * 30)
            print(f"IMAGE: {args.image}")
            print(f"RESULT: {transcription[0]}")
            print("-" * 30)
            
        except Exception as e:
            print(f"ERROR: {e}")
