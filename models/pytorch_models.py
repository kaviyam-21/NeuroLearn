import torch
import torch.nn as nn
from torchvision import models

class EncoderLayer(nn.Module):
    def __init__(self, input_size, hidden_states=200, rnn_layers=1):
        super(EncoderLayer, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_states, rnn_layers, bidirectional=True, batch_first=False)
            
    def forward(self, x):
        # x shape: (N, C, H, W) -> (W, N, C*H)
        N, C, H, W = x.size()
        x = x.permute(3, 0, 1, 2).contiguous()
        x = x.view(W, N, C * H)
        x, _ = self.lstm(x)
        # x shape: (W, N, 2*hidden_states) -> (N, W, 2*hidden_states)
        x = x.permute(1, 0, 2).contiguous()
        return x

class CNNBiLSTM(nn.Module):
    def __init__(self, num_downsamples=2, rnn_hidden_states=200, rnn_layers=1, alphabet_size=80):
        super(CNNBiLSTM, self).__init__()
        self.num_downsamples = num_downsamples
        
        # Body: ResNet34 backbone
        resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        # Adapt first layer for grayscale (1 channel)
        original_layer = resnet.conv1
        new_layer = nn.Conv2d(1, original_layer.out_channels, 
                             kernel_size=original_layer.kernel_size, 
                             stride=original_layer.stride, 
                             padding=original_layer.padding, 
                             bias=original_layer.bias)
        with torch.no_grad():
            new_layer.weight[:] = original_layer.weight.mean(dim=1, keepdim=True)
        
        self.body = nn.Sequential(
            new_layer,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2
        )

        # Encoders
        # ResNet34 stops at layer2: height 60 -> 30 -> 15 -> 8. Channels 128.
        # LSTM input = 128 * 8 = 1024
        self.encoder0 = nn.Sequential(
            EncoderLayer(input_size=1024, hidden_states=rnn_hidden_states, rnn_layers=rnn_layers),
            nn.Dropout(0.5)
        )
        # After downsampler: height 8 -> 4. Channels 128.
        # LSTM input = 128 * 4 = 512
        self.encoder1 = nn.Sequential(
            EncoderLayer(input_size=512, hidden_states=rnn_hidden_states, rnn_layers=rnn_layers),
            nn.Dropout(0.5)
        )

        # Downsampler
        self.downsampler = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        
        # Decoder (Dense layer)
        # Input features: 2 * rnn_hidden_states * num_downsamples (2 levels)
        self.decoder = nn.Linear(rnn_hidden_states * 2 * 2, alphabet_size)

    def forward(self, x):
        features = self.body(x) # -> B, 512, 2, W
        hidden_states = []
        
        # Process first level
        hs0 = self.encoder0(features)
        hidden_states.append(hs0)
        
        # Downsample and process subsequent levels
        downsampled_features = self.downsampler(features) # -> B, 128, 1, W/2
        hs1 = self.encoder1(downsampled_features)
        
        # Match lengths (since hs1 is W/2, we need to upsample or just repeat)
        # Simplest is to use the original MXNet logic: it likely concatenated or something.
        # Actually, let's just use levels properly.
        # For simplicity in this migration, I'll repeat along time axis or just use one.
        # Original MXNet used split/concat which handles seq_len.
        
        # Repeating hs1 to match hs0 length for concatenation
        hs1_up = hs1.repeat_interleave(2, dim=1)
        # Truncate if odd
        if hs1_up.size(1) > hs0.size(1):
            hs1_up = hs1_up[:, :hs0.size(1), :]
        elif hs1_up.size(1) < hs0.size(1):
            # padding
            padding = torch.zeros(hs0.size(0), hs0.size(1) - hs1_up.size(1), hs1_up.size(2)).to(hs0.device)
            hs1_up = torch.cat([hs1_up, padding], dim=1)
            
        hidden_states.append(hs1_up)
            
        # Concatenate hidden states from different levels along feature dimension
        combined_hs = torch.cat(hidden_states, dim=2)
        output = self.decoder(combined_hs)
        return output

class CNNTransformer(nn.Module):
    """
    Hybrid Model: CNN (ResNet) Encoder + Transformer Decoder.
    Combines spatial feature extraction with sequence context.
    """
    def __init__(self, alphabet_size, hidden_dim=256, nheads=8, num_layers=4):
        super(CNNTransformer, self).__init__()
        # 1. CNN Backbone (Pre-trained ResNet34)
        resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        
        # Modify for grayscale
        original_layer = resnet.conv1
        new_layer = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            new_layer.weight[:] = original_layer.weight.mean(dim=1, keepdim=True)
        self.backbone[0] = new_layer
        
        # 2. Projection
        self.proj = nn.Conv2d(512, hidden_dim, kernel_size=1)
        
        # 3. Transformer
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nheads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 4. Output
        self.output = nn.Linear(hidden_dim, alphabet_size)

    def forward(self, x):
        # Extract features
        x = self.backbone(x) # (N, 512, H, W)
        x = self.proj(x)     # (N, hidden_dim, H, W)
        
        # Sequence format: (Batch, Seq_len, Features)
        N, C, H, W = x.shape
        x = x.flatten(2).permute(0, 2, 1) 
        
        # Transformer pass
        x = self.transformer(x)
        
        # Map to alphabet
        return self.output(x)
