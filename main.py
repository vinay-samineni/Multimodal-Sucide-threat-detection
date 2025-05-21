import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer
import torchaudio
import torchvision.models as models

# -----------------
# Step 1: Dataset Loader (simplified)
# -----------------
class DAICWOZDataset(Dataset):
    def __init__(self, data_paths):
        self.data_paths = data_paths
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # Placeholder: load or define your video/audio loaders here

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        # Load raw text, audio waveform, video frames from disk here
        # Placeholder dummy data:
        text = "I am feeling okay today."  # example transcript
        audio_waveform = torch.randn(1, 16000)  # 1 sec fake audio, 16kHz
        video_frames = torch.randn(3, 224, 224)  # fake single video frame (C,H,W)

        # --- Text Preprocessing ---
        encoding = self.tokenizer(text, return_tensors='pt', padding='max_length', max_length=128, truncation=True)
        input_ids = encoding['input_ids'].squeeze(0)       # (seq_len)
        attention_mask = encoding['attention_mask'].squeeze(0)

        # --- Audio Preprocessing ---
        mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, n_mels=64)(audio_waveform)  # (1, 64, time)
        mel_spec = mel_spec.squeeze(0).transpose(0, 1)     # (time, 64)

        # --- Video Preprocessing ---
        # Normally multiple frames, here just 1 frame for example
        video = video_frames  # (3, 224, 224)

        # Placeholder risk label (0: Low, 1: Medium, 2: High)
        label = torch.tensor(1)

        return input_ids, attention_mask, mel_spec, video, label


# -----------------
# Step 2: Model Components
# -----------------

# Text feature extractor (BERT)
class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = output.last_hidden_state[:, 0, :]  # CLS token embedding (batch, 768)
        return cls_embedding


# Audio feature extractor (GRU over Mel Spectrogram)
class AudioEncoder(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=64):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)

    def forward(self, mel_spec):
        # mel_spec shape: (batch, time, 64)
        _, h_n = self.gru(mel_spec)  # h_n: (1, batch, hidden_dim)
        return h_n.squeeze(0)         # (batch, hidden_dim)


# Video feature extractor (CNN + LSTM)
class VideoEncoder(nn.Module):
    def __init__(self, lstm_hidden_dim=128):
        super().__init__()
        # Use pretrained ResNet18 for spatial features
        self.cnn = models.resnet18(pretrained=True)
        self.cnn.fc = nn.Identity()  # Remove final classification layer, output (batch, 512)

        # LSTM for temporal modeling (here simplified as one frame only)
        self.lstm = nn.LSTM(512, lstm_hidden_dim, batch_first=True)

    def forward(self, video_frames):
        # video_frames shape: (batch, seq_len, 3, 224, 224)
        batch_size, seq_len, C, H, W = video_frames.shape

        cnn_features = []
        for t in range(seq_len):
            frame = video_frames[:, t, :, :, :]  # (batch, 3, 224, 224)
            feat = self.cnn(frame)  # (batch, 512)
            cnn_features.append(feat)

        cnn_features = torch.stack(cnn_features, dim=1)  # (batch, seq_len, 512)
        _, (h_n, _) = self.lstm(cnn_features)            # h_n: (1, batch, lstm_hidden_dim)
        return h_n.squeeze(0)                             # (batch, lstm_hidden_dim)


# Cross-modal Attention Fusion Module
class CrossModalAttentionFusion(nn.Module):
    def __init__(self, input_dims=[768, 64, 128], proj_dim=256):
        super().__init__()
        self.text_proj = nn.Linear(input_dims[0], proj_dim)
        self.audio_proj = nn.Linear(input_dims[1], proj_dim)
        self.video_proj = nn.Linear(input_dims[2], proj_dim)

        self.attention_layer = nn.Sequential(
            nn.Linear(proj_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, text_feat, audio_feat, video_feat):
        # Project features
        text_proj = self.text_proj(text_feat)    # (batch, proj_dim)
        audio_proj = self.audio_proj(audio_feat)
        video_proj = self.video_proj(video_feat)

        # Stack modalities (batch, 3, proj_dim)
        modalities = torch.stack([text_proj, audio_proj, video_proj], dim=1)

        # Attention logits (batch, 3, 1)
        attn_logits = self.attention_layer(modalities)  
        attn_weights = torch.softmax(attn_logits, dim=1)  # (batch, 3, 1)

        # Weighted sum
        fused = torch.sum(attn_weights * modalities, dim=1)  # (batch, proj_dim)
        return fused, attn_weights.squeeze(-1)  # Return weights for explainability


# Risk Classification Layer (FCNN)
class RiskClassifier(nn.Module):
    def __init__(self, input_dim=256, num_classes=3):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.fc(x)  # logits


# -----------------
# Step 3: Full Model
# -----------------
class MultimodalRiskModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_encoder = TextEncoder()
        self.audio_encoder = AudioEncoder()
        self.video_encoder = VideoEncoder()
        self.fusion = CrossModalAttentionFusion()
        self.classifier = RiskClassifier()

    def forward(self, input_ids, attention_mask, mel_spec, video_frames):
        text_feat = self.text_encoder(input_ids, attention_mask)
        audio_feat = self.audio_encoder(mel_spec)
        video_feat = self.video_encoder(video_frames)
        fused, attn_weights = self.fusion(text_feat, audio_feat, video_feat)
        logits = self.classifier(fused)
        return logits, attn_weights


# -----------------
# Step 4: Usage Example (dummy batch)
# -----------------
if __name__ == "__main__":
    batch_size = 2
    seq_len = 1  # For video frames, use 1 here for simplicity

    # Dummy batch data
    input_ids = torch.randint(0, 30522, (batch_size, 128))          # BERT vocab size
    attention_mask = torch.ones(batch_size, 128)
    mel_spec = torch.randn(batch_size, 100, 64)                     # (batch, time, mel bins)
    video_frames = torch.randn(batch_size, seq_len, 3, 224, 224)    # (batch, seq_len, C,H,W)

    model = MultimodalRiskModel()
    logits, attn_weights = model(input_ids, attention_mask, mel_spec, video_frames)

    print("Logits shape:", logits.shape)          # (batch_size, 3)
    print("Attention weights:", attn_weights)    # (batch_size, 3)
