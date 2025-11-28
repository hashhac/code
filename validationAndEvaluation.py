import torch
import torch.nn as nn
import torchaudio
import os
from pathlib import Path
from tqdm import tqdm # Progress bar
from encodec.utils import convert_audio

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# 1. Checkpoint
CHECKPOINT_PATH = "Mask_Checkpoints/mask_model_ep29.pt" 

# 2. Input Folders (Where the noisy files live)
# Adjust these if your folder structure is slightly different
VALID_INPUT_DIR = Path("Candanza Data/cadenza_data/valid/unprocessed")
EVAL_INPUT_DIR = Path("Candanza Data/cadenza_data/eval/unprocessed") 

# 3. Output Folder
OUTPUT_BASE_DIR = Path("Enhanced_Output")

# 4. Audio Settings (Must match training)
N_FFT = 2048
HOP_LENGTH = 512
SAMPLE_RATE = 24000

# 5. Polishing (The "Punch" Factor)
APPLY_TRANSIENT_BOOST = True
BOOST_AMOUNT = 1.5 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==============================================================================
# MODEL ARCHITECTURE (Must match 'Magnum Opus' training exactly)
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
        pad_f = (8 - x.shape[3] % 8) % 8
        pad_t = (8 - x.shape[2] % 8) % 8
        x_padded = torch.nn.functional.pad(x, (0, pad_f, 0, pad_t))
        
        x1 = self.inc(x_padded)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.bot(x4)
        
        x = self.up1(x5)
        diffY = x4.size()[2] - x.size()[2]
        diffX = x4.size()[3] - x.size()[3]
        x = torch.nn.functional.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x4, x], dim=1)
        x = self.conv1(x)
        
        x = self.up2(x)
        diffY = x3.size()[2] - x.size()[2]
        diffX = x3.size()[3] - x.size()[3]
        x = torch.nn.functional.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x3, x], dim=1)
        x = self.conv2(x)
        
        x = self.up3(x)
        diffY = x1.size()[2] - x.size()[2]
        diffX = x1.size()[3] - x.size()[3]
        x = torch.nn.functional.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x1, x], dim=1)
        x = self.conv3(x)
        
        logits = self.outc(x)
        mask = torch.relu(logits)
        
        if pad_t > 0: mask = mask[:, :, :-pad_t, :]
        mask = mask[:, :, :, :self.n_freq_bins]
        return mask.squeeze(1)

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def load_model(path):
    print(f"Loading Checkpoint: {path}")
    model = ConvMaskingUNet(n_freq_bins=1025).to(device)
    
    # Handle state dict (sometimes wrapped in DDP or with metadata)
    checkpoint = torch.load(path, map_location=device)
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
        
    model.load_state_dict(state_dict)
    model.eval()
    return model

def add_punch(waveform, boost_amount=1.5):
    """Restores attack transients that might get smoothed by masking"""
    if boost_amount <= 1.0: return waveform
    
    # Simple envelope detection
    amp = waveform.abs()
    # Calculate derivative (change in volume)
    diff = torch.zeros_like(amp)
    diff[..., 1:] = amp[..., 1:] - amp[..., :-1]
    diff = torch.clamp(diff, min=0.0) # Only attacks
    
    # Create gain map based on attacks
    gain_map = 1.0 + (diff * boost_amount * 5.0)
    punched = waveform * gain_map
    
    # Safety Limiter (-0.95 to 0.95)
    return torch.clamp(punched, -0.95, 0.95)

def process_file(model, input_path, output_path):
    try:
        # 1. Load
        wav, sr = torchaudio.load(input_path)
        wav = convert_audio(wav, sr, SAMPLE_RATE, 1)
        wav = wav.to(device)
        
        # 2. STFT
        window = torch.hann_window(N_FFT).to(device)
        stft = torch.stft(wav, n_fft=N_FFT, hop_length=HOP_LENGTH, window=window, return_complex=True)
        mag = torch.abs(stft) + 1e-9
        phase = torch.angle(stft)
        
        # 3. Preprocess (Must match 'Magnum Opus' Dataset exactly)
        log_mag = torch.log(mag)
        log_mag = torch.clamp(log_mag, min=-20.0)
        
        # Per-Sample Stats
        mean = log_mag.mean()
        std = log_mag.std() + 1e-5
        model_input = (log_mag - mean) / std
        
        # Shape: [1, Freq, Time] -> [1, Time, Freq] (Batch=1)
        model_input = model_input.transpose(1, 2)
        
        # 4. Predict
        with torch.no_grad():
            mask = model(model_input) # Output [1, Time, Freq]
        
        # 5. Reconstruct
        mask = mask.transpose(1, 2) # Back to [1, Freq, Time]
        
        # Match lengths
        min_len = min(mask.shape[2], mag.shape[2])
        clean_mag = mag[..., :min_len] * mask[..., :min_len]
        clean_phase = phase[..., :min_len]
        
        clean_complex = torch.polar(clean_mag, clean_phase)
        clean_wav = torch.istft(clean_complex, n_fft=N_FFT, hop_length=HOP_LENGTH, window=window)
        
        # 6. Polish
        if APPLY_TRANSIENT_BOOST:
            clean_wav = add_punch(clean_wav, BOOST_AMOUNT)
            
        # 7. Save
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torchaudio.save(output_path, clean_wav.cpu(), SAMPLE_RATE)
        return True
        
    except Exception as e:
        print(f"FAILED on {input_path.name}: {e}")
        return False

# ==============================================================================
# MAIN LOOP
# ==============================================================================

def run_batch_inference():
    if not Path(CHECKPOINT_PATH).exists():
        print(f"Error: Checkpoint not found at {CHECKPOINT_PATH}")
        return

    model = load_model(CHECKPOINT_PATH)
    
    # List of folders to process
    tasks = []
    
    # Check Valid
    if VALID_INPUT_DIR.exists():
        tasks.append(("Valid", VALID_INPUT_DIR, OUTPUT_BASE_DIR / "valid"))
    else:
        print(f"Warning: Validation input not found at {VALID_INPUT_DIR}")
        
    # Check Eval
    if EVAL_INPUT_DIR.exists():
        tasks.append(("Eval", EVAL_INPUT_DIR, OUTPUT_BASE_DIR / "eval"))
    else:
        print(f"Warning: Evaluation input not found at {EVAL_INPUT_DIR}")
        
    if not tasks:
        print("Nothing to process!")
        return

    print(f"\n--- Starting Inference with {CHECKPOINT_PATH} ---")
    
    for task_name, in_dir, out_dir in tasks:
        print(f"\nProcessing {task_name} Set...")
        files = list(in_dir.glob("*.flac"))
        
        if not files:
            print(f"No files found in {in_dir}")
            continue
            
        print(f"Found {len(files)} files. Saving to {out_dir}")
        
        # Process with Progress Bar
        for file_path in tqdm(files):
            # Output name: same as input, or remove '_unproc' if you prefer
            out_name = file_path.name.replace("_unproc", "_enhanced")
            out_path = out_dir / out_name
            
            process_file(model, file_path, out_path)
            
    print("\n" + "="*60)
    print("ALL DONE. Sleep well! 💤")
    print("="*60)

if __name__ == "__main__":
    run_batch_inference()