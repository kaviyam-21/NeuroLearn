import time
import random
import os
import matplotlib.pyplot as plt
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from skimage import transform as skimage_tf
from skimage import exposure

from utilities.iam_dataset import IAMDataset, HF_IAMDataset, resize_image
from utilities.draw_text_on_image import draw_text_on_image
from models.pytorch_models import CNNBiLSTM

alphabet_encoding = r' !"#&\'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
alphabet_dict = {alphabet_encoding[i]:i for i in range(len(alphabet_encoding))}

def transform(image, label, max_seq_len):
    image = np.expand_dims(image, axis=0).astype(np.float32)
    if image.max() > 1:
        image = image/255.
    image = (image - 0.9425) / 0.1593
    
    label_encoded = np.zeros(max_seq_len, dtype=np.float32)-1
    i = 0
    for word in label:
        word = word.replace("&quot", r'"').replace("&amp", r'&').replace('";', '\"')
        for letter in word:
            if i < max_seq_len:
                label_encoded[i] = alphabet_dict.get(letter, 0)
                i += 1
    return image, label_encoded

def decode(prediction):
    results = []
    for word in prediction:
        result = []
        for i, index in enumerate(word):
            if i < len(word) - 1 and word[i] == word[i+1] and word[i] != -1:
                continue
            if index == len(alphabet_dict) or index == -1:
                continue
            else:
                result.append(alphabet_encoding[int(index)])
        results.append(result)
    return [''.join(word) for word in results]

def run_epoch(e, network, dataloader, optimizer, criterion, device, is_train):
    network.train() if is_train else network.eval()
    total_loss = 0.0
    for i, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)
        
        if is_train:
            optimizer.zero_grad()
            output = network(x)
            # CTCLoss expects (T, N, C)
            log_probs = output.permute(1, 0, 2).log_softmax(2)
            input_lengths = torch.full(size=(x.size(0),), fill_value=log_probs.size(0), dtype=torch.long).to(device)
            target_lengths = (y != -1).sum(dim=1).to(torch.long)
            targets = [row[row != -1] for row in y]
            targets_flat = torch.cat(targets).to(torch.long)
            
            loss = criterion(log_probs, targets_flat, input_lengths, target_lengths)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), 2.0)
            optimizer.step()
        else:
            with torch.no_grad():
                output = network(x)
                log_probs = output.permute(1, 0, 2).log_softmax(2)
                input_lengths = torch.full(size=(x.size(0),), fill_value=log_probs.size(0), dtype=torch.long).to(device)
                target_lengths = (y != -1).sum(dim=1).to(torch.long)
                targets = [row[row != -1] for row in y]
                targets_flat = torch.cat(targets).to(torch.long)
                loss = criterion(log_probs, targets_flat, input_lengths, target_lengths)
        
        total_loss += loss.item()
    return total_loss / len(dataloader) if len(dataloader) > 0 else 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gpu_id", default="0")
    parser.add_argument("-t", "--line_or_word", default="line")
    parser.add_argument("-e", "--epochs", type=int, default=121)
    parser.add_argument("-l", "--learning_rate", type=float, default=0.0001)
    parser.add_argument("-s", "--batch_size", type=int, default=64)
    parser.add_argument("--use_hf", action="store_true", help="Use Hugging Face Teklia/IAM-line dataset")
    parser.add_argument("--hf_train_split", type=str, default="train", help="HF dataset split for training (e.g. train[:50])")
    parser.add_argument("--hf_test_split", type=str, default="test", help="HF dataset split for testing (e.g. test[:10])")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() and args.gpu_id != "-1" else "cpu")
    max_seq_len = 100 if args.line_or_word == "line" else 32

    net = CNNBiLSTM(alphabet_size=len(alphabet_encoding)+1).to(device)
    optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)
    criterion = nn.CTCLoss(blank=len(alphabet_encoding), zero_infinity=True)

    if args.use_hf:
        print(f"Using Hugging Face dataset: Teklia/IAM-line (Splits: {args.hf_train_split}, {args.hf_test_split})")
        train_ds = HF_IAMDataset(split=args.hf_train_split)
        test_ds = HF_IAMDataset(split=args.hf_test_split)
    else:
        print("Using local dataset")
        train_ds = IAMDataset(args.line_or_word, train=True)
        test_ds = IAMDataset(args.line_or_word, train=False)

    from functools import partial
    train_ds.transform(partial(transform, max_seq_len=max_seq_len))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)

    test_ds.transform(partial(transform, max_seq_len=max_seq_len))
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    best_test_loss = float('inf')
    for e in range(args.epochs):
        train_loss = run_epoch(e, net, train_loader, optimizer, criterion, device, True)
        test_loss = run_epoch(e, net, test_loader, optimizer, criterion, device, False)
        print(f"Epoch {e}: Train Loss {train_loss:.4f}, Test Loss {test_loss:.4f}")
        
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(net.state_dict(), "best_model.pth")
            print(f"--> Saved best model with test loss: {best_test_loss:.4f}")
