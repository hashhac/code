import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json
import os
from encodec import EncodecModel
from encodec.utils import convert_audio

print("Initializing System: HEAVYWEIGHT CONFIGURATION")

# ==============================================================================
# PART 1: "MAX CAPACITY" CONFIGURATION
# ==============================================================================

# Upsampling Setup
HOP_LENGTH = 256           
N_FFT = 1024
print(f"Hop Length: {HOP_LENGTH}, FFT Size: {N_FFT} and D model is 1024")
# --- THE UPGRADE ---
N_MELS = 128               # High Res Input
D_MODEL = 1024              # WAS 256 -> DOUBLED (The "Brain Width")
N_HEAD = 8                 # WAS 4   -> DOUBLED (More attention context)
NUM_LAYERS = 12            # WAS 6   -> DOUBLED (Deeper reasoning)

# VRAM MANAGEMENT
BATCH_SIZE = 4             # REDUCED from 16 to fit the massive model
                           # If OOM (Out of Memory), set to 4.

LEARNING_RATE = 1e-4       # Standard for large Transformers
NUM_EPOCHS = 100
WARMUP_EPOCHS = 5          # Keep the identity warmup!

# Loss Weights (The "Token Pivot" Strategy)
LAMBDA_SEMANTIC = 10.0     # CRITICAL: This is the main goal now.
LAMBDA_PERCEPTUAL = 2.0    # Keep spectral shape correct
LAMBDA_MEL = 5.0           # Reduced slightly to let tokens take priority
LAMBDA_SIGNAL = 0.0        # OFF (Kills the buzz)
LAMBDA_PHASE = 0.0         # OFF
LAMBDA_HARMONIC = 0.0      # OFF

# Config
TARGET_BANDWIDTH = 24.0
MAX_GRAD_NORM = 1.0        # Tighter clipping for deeper networks

# Paths
DRIVE_BASE = Path("Candanza Data/cadenza_data")
METADATA_DIR = DRIVE_BASE / "metadata"
CHECKPOINT_DIR = Path("Checkpoints")
SAMPLE_DIR = Path("Training_Samples") 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"Model Dimensions: {D_MODEL}x{NUM_LAYERS} (Heads: {N_HEAD})")

# ==============================================================================
# PART 2 & 3: SETUP & LOSSES (Standard - No Changes Needed)
# ==============================================================================
# [Paste your standard EnCodec loading and Loss Classes here]
# ... (MultiResolutionSTFTLoss, MelSpectrogramLoss) ...
# I will skip pasting them to save space, they remain exactly the same.

# [RE-INSERT YOUR LOSS CLASSES HERE IF COPY-PASTING THE WHOLE FILE]
class MultiResolutionSTFTLoss(nn.Module):
    def __init__(self, fft_sizes=[1024, 2048, 512], hop_sizes=[120, 240, 50], win_lengths=[600, 1200, 240]):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths
    def forward(self, y_hat, y):
        total_loss = 0.0
        for fft_size, hop_size, win_length in zip(self.fft_sizes, self.hop_sizes, self.win_lengths):
            window = torch.hann_window(win_length).to(y.device)
            S_hat = torch.stft(y_hat, n_fft=fft_size, hop_length=hop_size, win_length=win_length, return_complex=True, window=window)
            S = torch.stft(y, n_fft=fft_size, hop_length=hop_size, win_length=win_length, return_complex=True, window=window)
            S_hat_mag = torch.abs(S_hat)
            S_mag = torch.abs(S)
            sc_loss = torch.norm(S_mag - S_hat_mag, p='fro') / (torch.norm(S_mag, p='fro') + 1e-7)
            mag_loss = F.l1_loss(torch.log(S_mag + 1e-7), torch.log(S_hat_mag + 1e-7))
            total_loss += sc_loss + mag_loss
        return total_loss / len(self.fft_sizes)

class MelSpectrogramLoss(nn.Module):
    def __init__(self, sample_rate=24000, n_fft=1024, hop_length=256, n_mels=80):
        super().__init__()
        self.mel_transform = T.MelSpectrogram(sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    def forward(self, y_hat, y):
        self.mel_transform = self.mel_transform.to(y.device)
        mel_hat = self.mel_transform(y_hat)
        mel_target = self.mel_transform(y)
        return F.l1_loss(torch.log(mel_hat + 1e-7), torch.log(mel_target + 1e-7))

# ==============================================================================
# PART 4: ARCHITECTURE (SCALED UP)
# ==============================================================================

class ResBlock1D(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=1):
        super().__init__()
        self.convs = nn.Sequential(
            nn.LeakyReLU(0.1),
            nn.Conv1d(channels, channels, kernel_size, dilation=dilation, padding=(kernel_size-1)*dilation//2),
            nn.LeakyReLU(0.1),
            nn.Conv1d(channels, channels, kernel_size, dilation=1, padding=(kernel_size-1)//2)
        )
    def forward(self, x): return x + self.convs(x)

class HiFiDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, upsample_factor):
        super().__init__()
        self.upsample = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=upsample_factor*2, stride=upsample_factor, padding=upsample_factor//2)
        self.res1 = ResBlock1D(out_channels, kernel_size=3, dilation=1)
        self.res2 = ResBlock1D(out_channels, kernel_size=7, dilation=3)
    def forward(self, x):
        x = self.upsample(x)
        x = self.res1(x)
        x = self.res2(x)
        return x

class SpeechEnhancementConformer(nn.Module):
    def __init__(self, input_features, num_tokens, num_codebooks, d_model, nhead, num_layers):
        super().__init__()
        self.num_codebooks = num_codebooks
        self.num_tokens = num_tokens

        # Encoder Projection
        self.input_proj = nn.Linear(input_features, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, 4000, d_model) * 0.02)
        
        # SCALED UP TRANSFORMER BACKBONE
        # dim_feedforward is usually 4x d_model. With d_model=512, this is 2048.
        # This is where the "Memory" of the model lives.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model * 4, 
            dropout=0.1, 
            batch_first=True,
            norm_first=True # Pre-Norm usually trains better for deep models
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Head 1: Tokens (The Primary Output)
        self.output_proj_logits = nn.Linear(d_model, num_codebooks * num_tokens)
        
        # Head 2: Audio Decoder (For Monitoring/Aux Loss)
        self.decoder_layers = nn.Sequential(
            HiFiDecoderBlock(d_model, d_model // 2, 8),
            HiFiDecoderBlock(d_model // 2, d_model // 4, 8),
            HiFiDecoderBlock(d_model // 4, d_model // 8, 4),
        )
        self.final_conv = nn.Sequential(
            nn.LeakyReLU(0.1),
            nn.Conv1d(d_model // 8, 1, kernel_size=7, padding=3),
            nn.Tanh()
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None: nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x_enc = self.input_proj(x)
        x_enc = x_enc + self.pos_encoder[:, :x_enc.size(1), :]
        x_enc = self.encoder(x_enc)
        
        logits = self.output_proj_logits(x_enc).view(x_enc.shape[0], x_enc.shape[1], self.num_codebooks, self.num_tokens)
        
        x_dec = x_enc.transpose(1, 2)
        x_dec = self.decoder_layers(x_dec)
        pred_wav = self.final_conv(x_dec)
        return logits, pred_wav

# ==============================================================================
# PART 5: DATASET (Standard)
# ==============================================================================
# (This remains exactly the same as the previous correct version)
# Ensure you copy the EncodecAudioDataset class from the previous code block
# Returns: specs, tokens, clean_wavs, unproc_wavs

# [RE-INSERT DATASET CLASS HERE]
class EncodecAudioDataset(Dataset):
    def __init__(self, metadata_path, unprocessed_dir, signals_dir, mel_transform, 
                 encodec_model, num_active_codebooks, device):
        self.mel_transform = mel_transform
        self.encodec_model = encodec_model
        self.num_active_codebooks = num_active_codebooks
        self.device = device 
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f: self.file_ids = [item['signal'] for item in json.load(f)]
        else: self.file_ids = []
        self.unprocessed_dir = Path(unprocessed_dir)
        self.signals_dir = Path(signals_dir)

    def __len__(self): return len(self.file_ids)

    def __getitem__(self, index):
        try:
            file_id = self.file_ids[index]
            unprocessed_path = self.unprocessed_dir / f"{file_id}_unproc.flac"
            clean_path = self.signals_dir / f"{file_id}.flac"
            
            unprocessed_wav, sr = torchaudio.load(unprocessed_path)
            unprocessed_wav = convert_audio(unprocessed_wav, sr, 24000, 1)
            
            unprocessed_wav_gpu = unprocessed_wav.to(self.device)
            mel_spec = self.mel_transform(unprocessed_wav_gpu)
            log_mel_spec = torch.log(mel_spec + 1e-5)
            log_mel_spec = (log_mel_spec - (-4.0)) / 4.0
            input_spectrogram = log_mel_spec.squeeze(0).transpose(0, 1)

            clean_wav, sr = torchaudio.load(clean_path)
            clean_wav = convert_audio(clean_wav, sr, 24000, 1) 
            
            clean_wav_gpu = clean_wav.to(self.device)
            with torch.no_grad():
                encoded_frames = self.encodec_model.encode(clean_wav_gpu.unsqueeze(0))
                all_tokens = encoded_frames[0][0]
                if all_tokens.dim() == 3: all_tokens = all_tokens.squeeze(0)
                target_tokens = all_tokens

            spec_len = input_spectrogram.shape[0]
            token_len = target_tokens.shape[1]
            min_frames = min(spec_len, token_len)
            
            input_spectrogram = input_spectrogram[:min_frames, :]
            target_tokens = target_tokens[:, :min_frames]
            
            expected_samples = min_frames * HOP_LENGTH
            if clean_wav.shape[-1] < expected_samples: clean_wav = F.pad(clean_wav, (0, expected_samples - clean_wav.shape[-1]))
            else: clean_wav = clean_wav[..., :expected_samples]
            if unprocessed_wav.shape[-1] < expected_samples: unprocessed_wav = F.pad(unprocessed_wav, (0, expected_samples - unprocessed_wav.shape[-1]))
            else: unprocessed_wav = unprocessed_wav[..., :expected_samples]
            
            return input_spectrogram.cpu(), target_tokens.cpu(), clean_wav, unprocessed_wav
        except Exception as e:
            print(f"Error: {e}")
            return None
            
def pad_collate(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0: return None, None, None, None
    (specs, tokens, clean_wavs, unproc_wavs) = zip(*batch)
    specs_padded = nn.utils.rnn.pad_sequence(specs, batch_first=True, padding_value=0.0)
    tokens_transposed = [t.transpose(0, 1) for t in tokens]
    tokens_padded = nn.utils.rnn.pad_sequence(tokens_transposed, batch_first=True, padding_value=0).permute(0, 2, 1)
    clean_transposed = [w.transpose(0, 1) for w in clean_wavs]
    clean_padded = nn.utils.rnn.pad_sequence(clean_transposed, batch_first=True, padding_value=0.0).permute(0, 2, 1)
    unproc_transposed = [w.transpose(0, 1) for w in unproc_wavs]
    unproc_padded = nn.utils.rnn.pad_sequence(unproc_transposed, batch_first=True, padding_value=0.0).permute(0, 2, 1)
    return specs_padded, tokens_padded, clean_padded, unproc_padded

# ==============================================================================
# PART 6: MAIN TRAINING (Standard, with Updated Config references)
# ==============================================================================
# [Paste main train() loop here - logic is same as previous, just update params]

def train():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    SAMPLE_DIR.mkdir(exist_ok=True)
    
    # SETUP ENCODEC
    print("\n--- Initializing EnCodec Model ---")
    encodec_model = EncodecModel.encodec_model_24khz()
    encodec_model.set_target_bandwidth(TARGET_BANDWIDTH)
    encodec_model.to(device)
    encodec_model.eval()
    
    ACTUAL_NUM_TOKENS = 1024 
    ACTUAL_NUM_CODEBOOKS = 32

    
    mel_transform = T.MelSpectrogram(sample_rate=24000, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS).to(device)
    
    train_dataset = EncodecAudioDataset(METADATA_DIR / "train_metadata.json", DRIVE_BASE / "train/unprocessed", DRIVE_BASE / "train/signals", mel_transform, encodec_model, ACTUAL_NUM_CODEBOOKS, device)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=pad_collate, drop_last=True)
    
    model = SpeechEnhancementConformer(N_MELS, ACTUAL_NUM_TOKENS, ACTUAL_NUM_CODEBOOKS, D_MODEL, N_HEAD, NUM_LAYERS).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    criterion_signal = nn.L1Loss()
    criterion_perceptual = MultiResolutionSTFTLoss().to(device)
    criterion_mel = MelSpectrogramLoss(sample_rate=24000, hop_length=HOP_LENGTH).to(device)
    
    print("\nStarting Training...")
    best_loss = float('inf')
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss_avg = 0
        
        # --- THE WARM-UP LOGIC ---
        if epoch < WARMUP_EPOCHS:
            if epoch == 0: print(f"--> STARTING WARM-UP: Identity Training (Input -> Input)")
            training_target = "identity"
        else:
            if epoch == WARMUP_EPOCHS: print(f"--> SWITCHING PHASE: Enhancement Training (Target -> Clean)")
            training_target = "clean"
        
        for batch_idx, (specs, tokens, clean_wavs, unproc_wavs) in enumerate(train_loader):
            if specs is None: continue
            
            specs = specs.to(device)
            tokens = tokens.to(device)
            clean_wavs = clean_wavs.to(device)
            unproc_wavs = unproc_wavs.to(device)
            
            if training_target == "identity":
                target_wavs = unproc_wavs
            else:
                target_wavs = clean_wavs
            
            optimizer.zero_grad()
            logits, pred_wavs = model(specs)
            
            min_len = min(pred_wavs.shape[-1], target_wavs.shape[-1])
            pred_wavs = pred_wavs[..., :min_len]
            target_wavs = target_wavs[..., :min_len]
            
            # Monitoring Loss (Decoder Head)
            loss_perc = criterion_perceptual(pred_wavs.squeeze(1), target_wavs.squeeze(1))
            loss_mel = criterion_mel(pred_wavs.squeeze(1), target_wavs.squeeze(1))
            
            # Primary Loss (Token Head)
            loss_sem = 0
            if training_target == "clean":
                for k in range(ACTUAL_NUM_CODEBOOKS):
                    loss_sem += F.cross_entropy(logits[:, :, k, :].reshape(-1, ACTUAL_NUM_TOKENS), tokens[:, k, :].reshape(-1))
                loss_sem /= ACTUAL_NUM_CODEBOOKS
            
            # Total Loss
            # During Identity phase: Focus purely on decoder physics
            if training_target == "identity":
                loss = (1.0 * loss_perc) + (10.0 * loss_mel)
            else:
                # During Clean phase: Focus on Tokens!
                loss = (LAMBDA_PERCEPTUAL * loss_perc) + (LAMBDA_MEL * loss_mel) + (LAMBDA_SEMANTIC * loss_sem)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            
            total_loss_avg += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Ep {epoch} [{batch_idx}] | {training_target.upper()} | Loss: {loss.item():.4f}")

        avg_loss = total_loss_avg / max(len(train_loader), 1)
        print(f"Epoch {epoch} Done. Avg Loss: {avg_loss:.4f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            model_config = {'input_features': N_MELS, 'num_tokens': ACTUAL_NUM_TOKENS, 'num_codebooks': ACTUAL_NUM_CODEBOOKS, 'd_model': D_MODEL, 'nhead': N_HEAD, 'num_layers': NUM_LAYERS}
            torch.save({
                'epoch': epoch, 'config': model_config, 
                'model_state_dict': model.state_dict(), 
                'optimizer_state_dict': optimizer.state_dict()
            }, CHECKPOINT_DIR / "best_model.pt")
            print("Saved Best Model.")
            
        with torch.no_grad():
            sample = pred_wavs[0].cpu()
            sample = sample / (sample.abs().max() + 1e-6)
            torchaudio.save(SAMPLE_DIR / f"ep{epoch}_{training_target}_sample.wav", sample, 24000)

if __name__ == '__main__':
    train()