import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import os
from pathlib import Path
import json
from encodec.utils import convert_audio

print("Initializing Stage 3: The Magnum Opus (Final Polish) YAYS! 🎉")

# ==============================================================================
# PART 1: CONFIGURATION
# ==============================================================================

N_FFT = 2048        
HOP_LENGTH = 512    
WIN_LENGTH = 2048
SAMPLE_RATE = 24000

BATCH_SIZE = 2      
ACCUM_STEPS = 16    # Virtual Batch Size = 32
LEARNING_RATE = 2e-4 
NUM_EPOCHS = 100
WARMUP_EPOCHS = 5   

DRIVE_BASE = Path("Candanza Data/cadenza_data")
METADATA_DIR = DRIVE_BASE / "metadata"
CHECKPOINT_DIR = Path("Mask_Checkpoints")
SAMPLE_DIR = Path("Mask_Samples")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==============================================================================
# PART 2: LOSS FUNCTION
# ==============================================================================

class AudioLoss(nn.Module):
    def __init__(self, n_freqs=1025, sample_rate=24000):
        super().__init__()
        
        # Create mel filterbank [1025, 128]
        mel_basis = torchaudio.functional.melscale_fbanks(
            n_freqs=n_freqs, 
            n_mels=128, 
            sample_rate=sample_rate,
            f_min=80, 
            f_max=8000 
        ).to(device)
        
        # Sum over mels (dim=1) to get weights per Frequency Bin [1025]
        freq_weights = mel_basis.sum(dim=1)
        freq_weights = freq_weights / (freq_weights.max() + 1e-8)
        
        # Reshape to [1, 1, 1025] for broadcasting
        self.register_buffer('freq_weights', freq_weights.unsqueeze(0).unsqueeze(0))
        
    def forward(self, est_mag, target_mag):
        # Normalize by target max per sample for scale invariance
        target_max = target_mag.amax(dim=(1, 2), keepdim=True) + 1e-8
        est_mag_norm = est_mag / target_max
        target_mag_norm = target_mag / target_max
        
        # 1. Weighted Magnitude Loss
        raw_mag_loss = F.l1_loss(est_mag_norm, target_mag_norm, reduction='none')
        mag_loss = (raw_mag_loss * self.freq_weights).mean()
        
        # 2. Log Magnitude Loss (log1p for stability)
        log_loss = F.l1_loss(torch.log1p(est_mag_norm), torch.log1p(target_mag_norm))
        
        # 3. Spectral Convergence
        sc_loss = torch.norm(target_mag_norm - est_mag_norm, p='fro') / \
                  (torch.norm(target_mag_norm, p='fro') + 1e-8)
        
        total_loss = mag_loss + 0.5 * log_loss + 0.5 * sc_loss
        
        return total_loss, {'mag': mag_loss.item(), 'log': log_loss.item(), 'sc': sc_loss.item()}

# ==============================================================================
# PART 3: DATASET & PADDING
# ==============================================================================

class MaskingDataset(torch.utils.data.Dataset):
    def __init__(self, metadata_path, unproc_dir, clean_dir, n_fft, hop, device):
        self.unproc_dir = Path(unproc_dir)
        self.clean_dir = Path(clean_dir)
        self.n_fft = n_fft
        self.hop = hop
        self.device = device
        self.window = torch.hann_window(n_fft).to(device)
        
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                self.file_ids = [item['signal'] for item in json.load(f)]
        else:
            self.file_ids = []
            
    def __len__(self): return len(self.file_ids)

    def __getitem__(self, index):
        try:
            file_id = self.file_ids[index]
            unproc_path = self.unproc_dir / f"{file_id}_unproc.flac"
            clean_path = self.clean_dir / f"{file_id}.flac"
            
            # Load & Convert
            unproc_wav, sr = torchaudio.load(unproc_path)
            unproc_wav = convert_audio(unproc_wav, sr, SAMPLE_RATE, 1) 
            clean_wav, sr = torchaudio.load(clean_path)
            clean_wav = convert_audio(clean_wav, sr, SAMPLE_RATE, 1)
            
            # Align
            min_len = min(unproc_wav.shape[-1], clean_wav.shape[-1])
            unproc_wav = unproc_wav[..., :min_len].to(self.device)
            clean_wav = clean_wav[..., :min_len].to(self.device)
            
            # STFT
            unproc_stft = torch.stft(unproc_wav, n_fft=self.n_fft, hop_length=self.hop, window=self.window, return_complex=True)
            clean_stft = torch.stft(clean_wav, n_fft=self.n_fft, hop_length=self.hop, window=self.window, return_complex=True)
            
            unproc_mag = torch.abs(unproc_stft) + 1e-9
            clean_mag = torch.abs(clean_stft) + 1e-9
            
            # Per-Sample Input Normalization
            unproc_log = torch.log(unproc_mag)
            unproc_log = torch.clamp(unproc_log, min=-20.0)
            
            mean = unproc_log.mean()
            std = unproc_log.std() + 1e-5
            model_input = (unproc_log - mean) / std
            
            # Transpose to [Time, Freq]
            model_input = model_input.squeeze(0).transpose(0, 1)
            
            return model_input, unproc_stft.squeeze(0), clean_mag.squeeze(0)
            
        except Exception as e:
            return None

def pad_collate(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0: return None, None, None
    
    (inputs, unproc_stfts, clean_mags) = zip(*batch)
    
    # Pad Inputs
    inputs_padded = nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=0.0)
    
    # Pad Complex STFT
    def pad_complex(c_list):
        transposed = [x.transpose(0, 1) for x in c_list] # [Time, Freq]
        max_len = max(x.shape[0] for x in transposed)
        padded_list = []
        for item in transposed:
            if item.shape[0] < max_len:
                diff = max_len - item.shape[0]
                try:
                    padded = F.pad(item, (0, 0, 0, diff))
                except:
                    # Fallback
                    padded = torch.complex(F.pad(item.real, (0, 0, 0, diff)), F.pad(item.imag, (0, 0, 0, diff)))
            else:
                padded = item
            padded_list.append(padded)
        return torch.stack(padded_list)

    unproc_stfts_padded = pad_complex(unproc_stfts)
    
    # Pad Mags (Transpose -> Pad -> Transpose back if needed, but we keep [Batch, Time, Freq])
    mags_transposed = [x.transpose(0, 1) for x in clean_mags]
    clean_mags_padded = nn.utils.rnn.pad_sequence(mags_transposed, batch_first=True, padding_value=0.0)
    
    # Returns:
    # Inputs: [Batch, Time, Freq]
    # STFTs: [Batch, Time, Freq] (Complex)
    # CleanMags: [Batch, Time, Freq]
    return inputs_padded, unproc_stfts_padded, clean_mags_padded

# ==============================================================================
# PART 4: MODEL
# ==============================================================================

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels), 
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
    def forward(self, x): return self.double_conv(x)

class ConvMaskingUNet(nn.Module):
    def __init__(self, n_freq_bins=1025):
        super().__init__()
        self.n_freq_bins = n_freq_bins
        
        # Learnable Input Normalization
        self.input_norm = nn.InstanceNorm2d(1, affine=True)
        
        self.inc = DoubleConv(1, 32)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(32, 64))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.bot = DoubleConv(256, 512)
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(256, 128)
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(96, 64)
        self.outc = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        x = x.unsqueeze(1) # [Batch, 1, Time, Freq]
        x = self.input_norm(x)
        
        pad_t = (8 - x.shape[2] % 8) % 8
        pad_f = (8 - x.shape[3] % 8) % 8
        x_padded = F.pad(x, (0, pad_f, 0, pad_t))
        
        x1 = self.inc(x_padded)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.bot(x4)
        
        x = self.up1(x5)
        x = self._cat(x, x4)
        x = self.conv1(x)
        x = self.up2(x)
        x = self._cat(x, x3)
        x = self.conv2(x)
        x = self.up3(x)
        x = self._cat(x, x1)
        x = self.conv3(x)
        
        logits = self.outc(x)
        mask = F.relu(logits) 
        
        if pad_t > 0: mask = mask[:, :, :-pad_t, :]
        mask = mask[:, :, :, :self.n_freq_bins]
        return mask.squeeze(1)
    
    def _cat(self, x, skip):
        diffY = skip.size(2) - x.size(2)
        diffX = skip.size(3) - x.size(3)
        x = F.pad(x, [diffX//2, diffX-diffX//2, diffY//2, diffY-diffY//2])
        return torch.cat([skip, x], dim=1)

# ==============================================================================
# PART 5: TRAINING LOOP
# ==============================================================================

def train():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(SAMPLE_DIR, exist_ok=True)
    
    print("--- Loading Dataset ---")
    dataset = MaskingDataset(METADATA_DIR / "train_metadata.json", DRIVE_BASE / "train/unprocessed", DRIVE_BASE / "train/signals", N_FFT, HOP_LENGTH, device)
    print(f"Dataset size: {len(dataset)}")
    
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=pad_collate, num_workers=0)
    
    model = ConvMaskingUNet(n_freq_bins=1025).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)
    loss_fn = AudioLoss(n_freqs=1025, sample_rate=SAMPLE_RATE)
    
    # Warmup + Cosine Scheduler
    from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
    warmup = LinearLR(optimizer, start_factor=0.1, total_iters=WARMUP_EPOCHS)
    cosine = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS - WARMUP_EPOCHS, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[WARMUP_EPOCHS])
    
    print(f"\n--- Starting Training ---")
    best_loss = float('inf')
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0
        batch_count = 0
        last_valid_batch = None
        optimizer.zero_grad()
        
        for batch_idx, batch_data in enumerate(loader):
            if batch_data is None: continue
            
            inputs, unproc_stft, clean_mags = batch_data
            inputs, clean_mags = inputs.to(device), clean_mags.to(device)
            unproc_stft = unproc_stft.to(device)
            
            # Forward
            predicted_mask = model(inputs)
            
            # Apply Mask to Magnitude
            # unproc_stft is [Batch, Time, Freq] due to collate
            unproc_mag = torch.abs(unproc_stft)
            est_mag = unproc_mag * predicted_mask
            
            # Loss
            loss, metrics = loss_fn(est_mag, clean_mags)
            loss = loss / ACCUM_STEPS
            loss.backward()
            
            # Step
            if (batch_idx + 1) % ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            actual_loss = loss.item() * ACCUM_STEPS
            epoch_loss += actual_loss
            batch_count += 1
            last_valid_batch = (inputs, unproc_stft)
            
            if batch_idx % 50 == 0:
                # FEATURE: Live Gradient Monitoring
                grad_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None: 
                        grad_norm += p.grad.data.norm(2).item() ** 2
                grad_norm = grad_norm ** 0.5
                
                print(f"Ep {epoch} [{batch_idx}] | Loss: {actual_loss:.4f} | "
                      f"Mag: {metrics['mag']:.3f} | Grad: {grad_norm:.3f}")

        if batch_count > 0:
            avg = epoch_loss / batch_count
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch} Done. Avg: {avg:.4f} | LR: {current_lr:.2e}")
            scheduler.step()
            
            if avg < best_loss:
                best_loss = avg
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg,
                }, CHECKPOINT_DIR / "best_model.pt")
                print(f"✓ New Best Saved: {avg:.4f}")
            
            if last_valid_batch is not None:
                try:
                    model.eval()
                    with torch.no_grad():
                        inputs, raw_stft = last_valid_batch
                        inp = inputs[0].unsqueeze(0)
                        pred_mask = model(inp)
                        
                        raw_sample = raw_stft[0].unsqueeze(0)
                        mag = torch.abs(raw_sample)
                        phase = torch.angle(raw_sample)
                        
                        min_t = min(pred_mask.shape[1], mag.shape[1])
                        # Apply Mask
                        clean_mag = mag[:, :min_t] * pred_mask[:, :min_t]
                        
                        # ISTFT (Reusing Noisy Phase)
                        # Need [Batch, Freq, Time] for ISTFT
                        complex_stft = torch.polar(clean_mag, phase[:, :min_t]).transpose(1, 2)
                        
                        clean_wav = torch.istft(
                            complex_stft, 
                            n_fft=N_FFT, hop_length=HOP_LENGTH, window=torch.hann_window(N_FFT).to(device)
                        )
                        torchaudio.save(SAMPLE_DIR / f"ep{epoch}_final_test.wav", clean_wav.cpu(), SAMPLE_RATE)
                    model.train()
                except Exception as e: print(f"Val Error: {e}")

            if (epoch + 1) % 5 == 0:
                torch.save(model.state_dict(), CHECKPOINT_DIR / f"mask_model_ep{epoch}.pt")

if __name__ == "__main__":
    train()