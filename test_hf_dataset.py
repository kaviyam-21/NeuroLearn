from datasets import load_dataset
import numpy as np

try:
    print("Loading Teklia/IAM-line dataset (split='train')...")
    # Using select to only download a tiny portion if possible, but datasets usually downloads metadata first
    ds = load_dataset("Teklia/IAM-line", split='train', trust_remote_code=True)
    
    # Get the first sample
    sample = ds[0]
    
    print("\nSample keys:", sample.keys())
    print("Image type:", type(sample['image']))
    
    # Check if 'text' or 'label' exists
    if 'text' in sample:
        print("Text sample:", sample['text'])
    elif 'text' in sample: # Some use transcription
        print("Transcription sample:", sample['text'])
        
    # Check image dimensions
    img = sample['image']
    print(f"Image mode: {img.mode}, Size: {img.size}")
    
except Exception as e:
    print(f"Error: {e}")
