import os
import shutil
import subprocess
from pathlib import Path
import sys
import torch 
import torchaudio
import torch.nn as nn
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from encodec.utils import convert_audio

# ==============================================================================
# CONFIGURATION
# ==============================================================================

TEST_LIMIT = 50 

# Paths
ROOT_DIR = Path(".") 
ORIGINAL_DATA_ROOT = ROOT_DIR / "Candanza Data"
TRAIN_INPUT_DIR = ORIGINAL_DATA_ROOT / "cadenza_data/train/unprocessed"
TRAIN_META_PATH = ORIGINAL_DATA_ROOT / "cadenza_data/metadata/train_metadata.json"

TESTING_DIR = ROOT_DIR / "testing"
ENHANCED_DIR = TESTING_DIR / "train_enhanced"
SHADOW_ROOT = ROOT_DIR / "Shadow_Dataset_ScoreMe"

CHECKPOINT_PATH = "Mask_Checkpoints/mask_model_ep29.pt" 

# Audio Settings
N_FFT = 2048
HOP_LENGTH = 512
SAMPLE_RATE = 24000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"--- INITIALIZING SCORE ME (Testing on {TEST_LIMIT} files) ---")

# ==============================================================================
# 1. MODEL ARCHITECTURE
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
        x = x.unsqueeze(1)
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
        x = torch.nn.functional.pad(x, [diffX//2, diffX-diffX//2, diffY//2, diffY-diffY//2])
        x = torch.cat([x4, x], dim=1)
        x = self.conv1(x)
        x = self.up2(x)
        diffY = x3.size()[2] - x.size()[2]
        diffX = x3.size()[3] - x.size()[3]
        x = torch.nn.functional.pad(x, [diffX//2, diffX-diffX//2, diffY//2, diffY-diffY//2])
        x = torch.cat([x3, x], dim=1)
        x = self.conv2(x)
        x = self.up3(x)
        diffY = x1.size()[2] - x.size()[2]
        diffX = x1.size()[3] - x.size()[3]
        x = torch.nn.functional.pad(x, [diffX//2, diffX-diffX//2, diffY//2, diffY-diffY//2])
        x = torch.cat([x1, x], dim=1)
        x = self.conv3(x)
        logits = self.outc(x)
        mask = torch.relu(logits)
        if pad_t > 0: mask = mask[:, :, :-pad_t, :]
        mask = mask[:, :, :, :self.n_freq_bins]
        return mask.squeeze(1)

# ==============================================================================
# 2. INFERENCE
# ==============================================================================

def run_inference():
    print("\n--- 1. Enhancing Training Data ---")
    
    if not TRAIN_INPUT_DIR.exists():
        print(f"❌ Error: Input dir not found: {TRAIN_INPUT_DIR}")
        return False
        
    ENHANCED_DIR.mkdir(parents=True, exist_ok=True)
    
    model = ConvMaskingUNet(n_freq_bins=1025).to(device)
    # Using False for now to be safe with older checkpoints
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()
    
    all_files = list(TRAIN_INPUT_DIR.glob("*.flac"))
    if TEST_LIMIT:
        import random
        random.seed(42) 
        random.shuffle(all_files)
        files_to_process = all_files[:TEST_LIMIT]
    else:
        files_to_process = all_files
        
    print(f"Processing {len(files_to_process)} files...")
    processed_count = 0
    
    with torch.no_grad():
        for f_path in tqdm(files_to_process):
            try:
                out_name = f_path.name.replace("_unproc", "")
                dest_path = ENHANCED_DIR / out_name
                
                # Skip if already exists (saves time on reruns)
                if dest_path.exists():
                    processed_count += 1
                    continue

                wav, sr = torchaudio.load(f_path)
                wav = convert_audio(wav, sr, SAMPLE_RATE, 1).to(device)
                
                window = torch.hann_window(N_FFT).to(device)
                stft = torch.stft(wav, n_fft=N_FFT, hop_length=HOP_LENGTH, window=window, return_complex=True)
                mag = torch.abs(stft) + 1e-9
                phase = torch.angle(stft)
                
                log_mag = torch.log(mag)
                log_mag = torch.clamp(log_mag, min=-20.0)
                mean = log_mag.mean()
                std = log_mag.std() + 1e-5
                model_input = (log_mag - mean) / std
                model_input = model_input.transpose(1, 2)
                
                mask = model(model_input).transpose(1, 2)
                
                min_len = min(mask.shape[2], mag.shape[2])
                clean_mag = mag[..., :min_len] * mask[..., :min_len]
                clean_complex = torch.polar(clean_mag, phase[..., :min_len])
                
                clean_wav = torch.istft(clean_complex, n_fft=N_FFT, hop_length=HOP_LENGTH, window=window)
                
                # Polish
                boost = 1.5
                amp = clean_wav.abs()
                diff = torch.zeros_like(amp)
                diff[..., 1:] = amp[..., 1:] - amp[..., :-1]
                diff = torch.clamp(diff, min=0.0)
                gain = 1.0 + (diff * boost * 5.0)
                clean_wav = clean_wav * gain
                clean_wav = torch.clamp(clean_wav, -0.95, 0.95)
                
                stereo_wav = torch.cat([clean_wav, clean_wav], dim=0)
                torchaudio.save(dest_path, stereo_wav.cpu(), SAMPLE_RATE)
                processed_count += 1
                
            except Exception as e:
                print(f"Failed {f_path.name}: {e}")
                
    return processed_count > 0

# ==============================================================================
# 3. WHISPER SCORING (With Metadata Filtering)
# ==============================================================================

def run_whisper_scoring():
    print("\n--- 2. Running Official Whisper Scorer ---")
    
    if SHADOW_ROOT.exists(): shutil.rmtree(SHADOW_ROOT)
    
    signals_dir = SHADOW_ROOT / "cadenza_data/audio/train/signals"
    meta_dir = SHADOW_ROOT / "cadenza_data/metadata"
    signals_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Copy Files
    generated_files = list(ENHANCED_DIR.glob("*.flac"))
    print(f"Copying {len(generated_files)} enhanced files...")
    for f in generated_files:
        shutil.copy(f, signals_dir / f.name)
        
    # 2. FILTER METADATA (The Critical Fix)
    # We only want metadata entries for the 50 files we actually generated.
    print("Filtering metadata to match generated files...")
    
    if not TRAIN_META_PATH.exists():
        print("❌ Original Metadata missing!")
        return False
        
    # Get IDs of generated files (filename without extension)
    generated_ids = set(f.stem for f in generated_files)
    
    with open(TRAIN_META_PATH) as f:
        full_meta = json.load(f)
        
    # Filter list
    filtered_meta = [record for record in full_meta if record['signal'] in generated_ids]
    
    print(f"Filtered Metadata: {len(filtered_meta)} records (Original: {len(full_meta)})")
    
    # Save to Shadow Dataset
    with open(meta_dir / "train_metadata.json", 'w') as f:
        json.dump(filtered_meta, f)
        
    # 3. Run Compute Whisper
    compute_script = list(ROOT_DIR.rglob("compute_whisper.py"))[0]
    contractions_file = list(ROOT_DIR.rglob("contractions.csv"))[0]
    clarity_lib = list(ROOT_DIR.rglob("clarity/utils"))[0].parent.parent
    
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{clarity_lib}:{env.get('PYTHONPATH', '')}"
    
    jsonl_out = "cadenza_data.train.whisper.jsonl"
    if Path(jsonl_out).exists(): os.remove(jsonl_out)
    
    cmd = [
        "python", str(compute_script),
        f"data.cadenza_data_root={SHADOW_ROOT.absolute()}",
        "split=train", 
        "baseline=whisper", "baseline.system=whisper", "hydra.run.dir=.",
        "baseline.reference=processed",
        f"baseline.contractions_file={contractions_file.absolute()}"
    ]
    
    subprocess.run(cmd, check=True, env=env)
    return True

# ==============================================================================
# 4. SCORE CALCULATION
# ==============================================================================

def calculate_final_score():
    print("\n--- 3. Calculating RMSE Score ---")
    
    # 1. Load Ground Truth from the Shadow/Filtered Metadata
    meta_path = SHADOW_ROOT / "cadenza_data/metadata/train_metadata.json"
    with open(meta_path) as f:
        meta_list = json.load(f)
    
    # Load Truth
    truth_map = {item['signal']: float(item['correctness']) for item in meta_list}
    
    # 2. Load Predictions
    jsonl_path = Path("cadenza_data.train.whisper.jsonl")
    if not jsonl_path.exists():
        print("❌ Whisper output file not found.")
        return

    predictions = []
    truths = []
    
    print(f"Reading results from {jsonl_path}...")
    
    with open(jsonl_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            sig_id = data['signal']
            
            # Prediction: Whisper Raw (0-1) -> Scale to 0-100
            pred_score = data['whisper'] * 100.0
            
            if sig_id in truth_map:
                true_score = truth_map[sig_id]
                
                # --- THE FIX: AUTO-SCALE GROUND TRUTH ---
                # If Ground Truth is fractional (e.g. 0.45), scale to percentage (45.0)
                # We check if the max truth in the batch is <= 1.0 to decide
                # (Done per item here for simplicity, assuming < 1.05 means fraction)
                if true_score <= 1.05 and true_score > 0:
                    true_score *= 100.0
                
                predictions.append(pred_score)
                truths.append(true_score)
    
    if not predictions:
        print("No matching records found!")
        return
        
    y_pred = np.array(predictions)
    y_true = np.array(truths)
    
    # Metrics
    rmse = np.sqrt(np.mean((y_pred - y_true)**2))
    
    print("\n" + "="*60)
    print(f"🧪 INTERNAL VALIDATION RESULTS ({len(predictions)} files)")
    print("="*60)
    print(f"AVG PRED:   {np.mean(y_pred):.2f}%")
    print(f"AVG TRUTH:  {np.mean(y_true):.2f}%")
    print("-" * 30)
    print(f"RMSE SCORE: {rmse:.4f}")
    print("="*60)

if __name__ == "__main__":
    if run_inference():
        if run_whisper_scoring():
            calculate_final_score()