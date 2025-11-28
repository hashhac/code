import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
from pathlib import Path
import os
import json
import shutil
from encodec import EncodecModel
from encodec.utils import convert_audio

# ==============================================================================
# PART 1: MODEL ARCHITECTURE (MUST MATCH TRAINING EXACTLY)
# ==============================================================================

class ResidualUnit(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=1):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.layers = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv1d(channels, channels, kernel_size, dilation=dilation, padding=padding),
            nn.LeakyReLU(0.2),
            nn.Conv1d(channels, channels, kernel_size, dilation=1, padding=(kernel_size - 1) // 2)
        )
    def forward(self, x):
        return x + self.layers(x)

class SpeechEnhancementConformer(nn.Module):
    def __init__(self, input_features, num_tokens, num_codebooks, d_model, nhead, num_layers):
        super().__init__()
        self.num_codebooks = num_codebooks
        self.num_tokens = num_tokens

        # Encoder
        self.input_proj = nn.Linear(input_features, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, 4000, d_model) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Head 1: Semantic
        self.output_proj_logits = nn.Linear(d_model, num_codebooks * num_tokens)
        
        # Head 2: Waveform Decoder (The "New" Architecture)
        self.wav_decoder = nn.Sequential(
            nn.ConvTranspose1d(d_model, d_model // 2, kernel_size=16, stride=8, padding=4),
            ResidualUnit(d_model // 2),
            nn.ConvTranspose1d(d_model // 2, d_model // 4, kernel_size=16, stride=8, padding=4),
            ResidualUnit(d_model // 4),
            nn.ConvTranspose1d(d_model // 4, d_model // 8, kernel_size=8, stride=4, padding=2),
            ResidualUnit(d_model // 8),
            nn.LeakyReLU(0.2),
            nn.Conv1d(d_model // 8, 1, kernel_size=7, padding=3),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = x + self.pos_encoder[:, :x.size(1), :]
        x = self.encoder(x)
        logits = self.output_proj_logits(x).view(x.shape[0], x.shape[1], self.num_codebooks, self.num_tokens)
        predicted_wav = self.wav_decoder(x.transpose(1, 2))
        return logits, predicted_wav

# ==============================================================================
# PART 2: INFERENCE LOGIC
# ==============================================================================

def load_model(checkpoint_path, device):
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 1. Load Config (Vital for reconstructing model)
    if 'config' not in checkpoint:
        raise ValueError("Checkpoint does not contain 'config'. Cannot reconstruct model.")
    
    config = checkpoint['config']
    print("Model Config Found:")
    print(json.dumps(config, indent=2))
    
    # 2. Initialize Model
    model = SpeechEnhancementConformer(
        input_features=config['input_features'],
        num_tokens=config['num_tokens'],
        num_codebooks=config['num_codebooks'],
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers']
    ).to(device)
    
    # 3. Clean State Dict (Handle torch.compile prefix)
    state_dict = checkpoint['model_state_dict']
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("_orig_mod.", "") # Remove compile prefix
        new_state_dict[new_key] = value
        
    model.load_state_dict(new_state_dict)
    model.eval()
    
    return model, config

def run_inference(checkpoint_path, input_audio_path, output_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # 1. Load Model
    model, config = load_model(checkpoint_path, device)
    
    # 2. Prepare Audio
    print(f"\nProcessing: {input_audio_path}")
    wav, sr = torchaudio.load(input_audio_path)
    # Convert to 24kHz Mono
    wav = convert_audio(wav, sr, 24000, 1).to(device)
    
    # 3. Create Spectrogram (Exact same logic as training)
    mel_transform = T.MelSpectrogram(
        sample_rate=24000, 
        n_fft=1024, 
        hop_length=256, 
        n_mels=config['input_features'] # Uses 128 from config
    ).to(device)
    
    mel_spec = mel_transform(wav)
    
    # --- PREPROCESSING (Must match training!) ---
    log_mel = torch.log(mel_spec + 1e-5)
    
    # Apply Normalization (Mean -4, Std 4)
    norm_mel = (log_mel - (-4.0)) / 4.0
    # --------------------------------------------
    
    # Shape: [1, time, n_mels]
    input_tensor = norm_mel.squeeze(0).transpose(0, 1).unsqueeze(0)
    
    # 4. Run Inference
    with torch.no_grad():
        logits, predicted_wav = model(input_tensor)
        
    # 5. Post-Process
    output_wav = predicted_wav.squeeze(0).cpu() # [1, time]
    
    # Check output stats
    print(f"Output Stats -> Max: {output_wav.abs().max():.4f}, Mean: {output_wav.mean():.4f}")
    
    # Save
    filename = Path(input_audio_path).stem
    save_path = output_dir / f"{filename}_cleaned.wav"
    torchaudio.save(save_path, output_wav, 24000)
    print(f"Saved to: {save_path}")

# ==============================================================================
# RUN IT
# ==============================================================================

if __name__ == "__main__":
    # --- SETTINGS ---
    CHECKPOINT_FILE = "Checkpoints/best_model.pt" # Points to the saved best model
    
    # Pick a specific file to test
    TEST_FILE = "Candanza Data/cadenza_data/valid/unprocessed/0a01ea4ed560ae24f7c27806_unproc.flac" 
    OUTPUT_FOLDER = "Inference_Results"
    
    if os.path.exists(CHECKPOINT_FILE) and os.path.exists(TEST_FILE):
        run_inference(CHECKPOINT_FILE, TEST_FILE, OUTPUT_FOLDER)
    else:
        print("Error: Checkpoint or Test File not found.")
        print(f"Checkpoint: {CHECKPOINT_FILE}")
        print(f"File: {TEST_FILE}")