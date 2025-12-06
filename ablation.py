import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import json
from encodec.utils import convert_audio
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

print("🔬 Initializing Comprehensive Ablation & Analysis Suite")

# ==============================================================================
# CONFIGURATION
# ==============================================================================

CHECKPOINT_PATH = "Mask_Checkpoints/best_model.pt"
TEST_AUDIO_DIR = Path("Candanza Data/cadenza_data/valid/unprocessed")
CLEAN_AUDIO_DIR = Path("Candanza Data/cadenza_data/valid/signals")
OUTPUT_DIR = Path("Ablation_Results")
N_TEST_SAMPLES = 10  # Number of files to analyze

N_FFT = 2048
HOP_LENGTH = 512
SAMPLE_RATE = 24000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Create output structure
(OUTPUT_DIR / "plots").mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "audio_samples").mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "data").mkdir(parents=True, exist_ok=True)

# ==============================================================================
# MODEL ARCHITECTURE (EXACT MATCH)
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

    def forward(self, x, return_features=False):
        """
        If return_features=True, returns intermediate activations for analysis
        """
        features = {}
        
        x = x.unsqueeze(1)
        features['input'] = x.clone()
        
        x = self.input_norm(x)
        features['normalized_input'] = x.clone()
        
        pad_f = (8 - x.shape[3] % 8) % 8
        pad_t = (8 - x.shape[2] % 8) % 8
        x = F.pad(x, (0, pad_f, 0, pad_t))
        
        # Encoder
        x1 = self.inc(x)
        features['enc_1'] = x1.clone()
        x2 = self.down1(x1)
        features['enc_2'] = x2.clone()
        x3 = self.down2(x2)
        features['enc_3'] = x3.clone()
        x4 = self.down3(x3)
        features['enc_4'] = x4.clone()
        x5 = self.bot(x4)
        features['bottleneck'] = x5.clone()
        
        # Decoder
        x = self.up1(x5)
        x = self._pad_and_cat(x, x4)
        x = self.conv1(x)
        features['dec_1'] = x.clone()
        
        x = self.up2(x)
        x = self._pad_and_cat(x, x3)
        x = self.conv2(x)
        features['dec_2'] = x.clone()
        
        x = self.up3(x)
        x = self._pad_and_cat(x, x1)
        x = self.conv3(x)
        features['dec_3'] = x.clone()
        
        logits = self.outc(x)
        features['logits'] = logits.clone()
        mask = F.relu(logits)
        
        if pad_t > 0: mask = mask[:, :, :-pad_t, :]
        mask = mask[:, :, :, :self.n_freq_bins]
        
        if return_features:
            return mask.squeeze(1), features
        return mask.squeeze(1)
    
    def _pad_and_cat(self, x, skip):
        diffY = skip.size(2) - x.size(2)
        diffX = skip.size(3) - x.size(3)
        x = F.pad(x, [diffX//2, diffX-diffX//2, diffY//2, diffY-diffY//2])
        return torch.cat([skip, x], dim=1)

# ==============================================================================
# AUDIO PROCESSING UTILITIES
# ==============================================================================

def load_audio_pair(noisy_path, clean_path):
    """Load and align noisy/clean audio pair"""
    noisy_wav, sr = torchaudio.load(noisy_path)
    noisy_wav = convert_audio(noisy_wav, sr, SAMPLE_RATE, 1).to(device)
    
    clean_wav, sr = torchaudio.load(clean_path)
    clean_wav = convert_audio(clean_wav, sr, SAMPLE_RATE, 1).to(device)
    
    # Align
    min_len = min(noisy_wav.shape[-1], clean_wav.shape[-1])
    return noisy_wav[..., :min_len], clean_wav[..., :min_len]

def preprocess_for_model(mag):
    """Apply same preprocessing as training"""
    log_mag = torch.log(mag + 1e-9)
    log_mag = torch.clamp(log_mag, min=-20.0)
    mean = log_mag.mean()
    std = log_mag.std() + 1e-5
    return (log_mag - mean) / std

def compute_stft(wav):
    """Compute STFT with training parameters"""
    window = torch.hann_window(N_FFT).to(device)
    stft = torch.stft(wav, n_fft=N_FFT, hop_length=HOP_LENGTH, 
                      window=window, return_complex=True)
    return stft

def stft_to_audio(stft_complex):
    """Convert STFT back to waveform"""
    window = torch.hann_window(N_FFT).to(device)
    return torch.istft(stft_complex, n_fft=N_FFT, hop_length=HOP_LENGTH, window=window)


def _ensure_batch_time_freq(x: torch.Tensor) -> torch.Tensor:
    """Ensure tensor has shape [Batch, Time, Freq].
    Accepts inputs of shapes:
      - [Time, Freq] -> unsqueeze(0)
      - [1, Time, Freq] -> leave as-is
      - [Batch, 1, Time, Freq] -> squeeze channel dim
      - [Batch, Time, Freq] -> leave as-is
    This prevents double-unsqueezing that produced 5D tensors.
    """
    if x is None:
        return x
    if not isinstance(x, torch.Tensor):
        return x
    if x.dim() == 2:
        return x.unsqueeze(0)
    if x.dim() == 3:
        return x
    if x.dim() == 4:
        # assume [batch, channel, time, freq]
        if x.size(1) == 1:
            return x.squeeze(1)
        # if channels >1, collapse channels into batch (rare) by reshaping
        b, c, t, f = x.shape
        return x.view(b * c, t, f)
    return x

# ==============================================================================
# METRICS
# ==============================================================================

def compute_snr(clean, enhanced):
    """Signal-to-Noise Ratio in dB"""
    # align lengths on last dimension to avoid size mismatch
    if clean is None or enhanced is None:
        return float('nan')
    min_len = min(clean.shape[-1], enhanced.shape[-1])
    clean_a = clean[..., :min_len]
    enhanced_a = enhanced[..., :min_len]
    noise = enhanced_a - clean_a
    signal_power = (clean_a ** 2).mean()
    noise_power = (noise ** 2).mean()
    return 10 * torch.log10(signal_power / (noise_power + 1e-8))

def compute_si_sdr(reference, estimate):
    """Scale-Invariant Signal-to-Distortion Ratio"""
    # align lengths on last dimension
    if reference is None or estimate is None:
        return float('nan')
    min_len = min(reference.shape[-1], estimate.shape[-1])
    ref = reference[..., :min_len]
    est = estimate[..., :min_len]

    ref = ref - ref.mean(dim=-1, keepdim=True)
    est = est - est.mean(dim=-1, keepdim=True)

    # compute scalar alpha per example (supports batched tensors)
    # sum over last dim
    numerator = (est * ref).sum(dim=-1)
    denom = (ref ** 2).sum(dim=-1) + 1e-8
    alpha = numerator / denom
    projection = alpha.unsqueeze(-1) * ref
    noise = est - projection

    ratio = (projection ** 2).sum(dim=-1) / ((noise ** 2).sum(dim=-1) + 1e-8)
    return 10 * torch.log10(ratio)

def spectral_distance(spec1, spec2):
    """L2 distance between spectrograms"""
    mag1 = torch.abs(spec1)
    mag2 = torch.abs(spec2)
    return torch.norm(mag1 - mag2) / torch.norm(mag1)

# ==============================================================================
# ANALYSIS 1: MODEL WEIGHT DISTRIBUTION
# ==============================================================================

def analyze_weight_distributions(model):
    """Visualize weight distributions across layers"""
    print("\n📊 Analyzing weight distributions...")
    
    fig, axes = plt.subplots(3, 4, figsize=(20, 12))
    axes = axes.flatten()
    
    layer_weights = []
    layer_names = []
    
    for name, param in model.named_parameters():
        if 'weight' in name and len(param.shape) >= 2:
            weights = param.data.cpu().numpy().flatten()
            layer_weights.append(weights)
            layer_names.append(name.split('.')[0])
    
    # Plot distributions
    for idx, (weights, name) in enumerate(zip(layer_weights[:12], layer_names[:12])):
        axes[idx].hist(weights, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        axes[idx].set_title(f'{name}\nμ={weights.mean():.4f}, σ={weights.std():.4f}')
        axes[idx].set_xlabel('Weight Value')
        axes[idx].set_ylabel('Count')
        axes[idx].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "plots" / "weight_distributions.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Weight magnitude heatmap
    fig, ax = plt.subplots(figsize=(14, 8))
    weight_mags = [np.abs(w).mean() for w in layer_weights]
    weight_stds = [np.abs(w).std() for w in layer_weights]
    
    x = np.arange(len(weight_mags))
    ax.bar(x, weight_mags, yerr=weight_stds, alpha=0.7, color='coral', edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels(layer_names, rotation=45, ha='right')
    ax.set_ylabel('Mean Absolute Weight')
    ax.set_title('Weight Magnitudes Across Layers')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "plots" / "weight_magnitudes.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Weight analysis complete")

# ==============================================================================
# ANALYSIS 2: FEATURE ACTIVATION PATTERNS
# ==============================================================================

def analyze_feature_activations(model, test_files):
    """Analyze activation patterns through the network"""
    print("\n🧠 Analyzing feature activations...")
    
    model.eval()
    activation_stats = defaultdict(list)
    
    with torch.no_grad():
        for audio_path in tqdm(test_files[:5], desc="Processing samples"):
            # Load
            wav, sr = torchaudio.load(audio_path)
            wav = convert_audio(wav, sr, SAMPLE_RATE, 1).to(device)
            
            # Process
            stft = compute_stft(wav)
            mag = torch.abs(stft)
            model_input = preprocess_for_model(mag).transpose(1, 2)
            
            # Ensure shape [Batch, Time, Freq] and forward with features
            model_input = _ensure_batch_time_freq(model_input)
            _, features = model(model_input, return_features=True)
            
            # Collect statistics
            for layer_name, activation in features.items():
                activation_stats[layer_name].append({
                    'mean': activation.mean().item(),
                    'std': activation.std().item(),
                    'max': activation.max().item(),
                    'sparsity': (activation == 0).float().mean().item()
                })
    
    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    layers = list(activation_stats.keys())
    means = [np.mean([s['mean'] for s in activation_stats[l]]) for l in layers]
    stds = [np.mean([s['std'] for s in activation_stats[l]]) for l in layers]
    maxs = [np.mean([s['max'] for s in activation_stats[l]]) for l in layers]
    sparsity = [np.mean([s['sparsity'] for s in activation_stats[l]]) for l in layers]
    
    # Mean activations
    axes[0, 0].bar(range(len(layers)), means, color='skyblue', edgecolor='black')
    axes[0, 0].set_xticks(range(len(layers)))
    axes[0, 0].set_xticklabels(layers, rotation=45, ha='right')
    axes[0, 0].set_ylabel('Mean Activation')
    axes[0, 0].set_title('Average Activation Levels')
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # Std deviations
    axes[0, 1].bar(range(len(layers)), stds, color='lightcoral', edgecolor='black')
    axes[0, 1].set_xticks(range(len(layers)))
    axes[0, 1].set_xticklabels(layers, rotation=45, ha='right')
    axes[0, 1].set_ylabel('Std Deviation')
    axes[0, 1].set_title('Activation Variability')
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # Max activations
    axes[1, 0].bar(range(len(layers)), maxs, color='lightgreen', edgecolor='black')
    axes[1, 0].set_xticks(range(len(layers)))
    axes[1, 0].set_xticklabels(layers, rotation=45, ha='right')
    axes[1, 0].set_ylabel('Max Activation')
    axes[1, 0].set_title('Peak Activation Values')
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # Sparsity
    axes[1, 1].bar(range(len(layers)), sparsity, color='plum', edgecolor='black')
    axes[1, 1].set_xticks(range(len(layers)))
    axes[1, 1].set_xticklabels(layers, rotation=45, ha='right')
    axes[1, 1].set_ylabel('Sparsity (% zeros)')
    axes[1, 1].set_title('Activation Sparsity')
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "plots" / "activation_patterns.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Activation analysis complete")

# ==============================================================================
# ANALYSIS 3: MASK BEHAVIOR ANALYSIS
# ==============================================================================

def analyze_mask_behavior(model, test_files):
    """Analyze what the mask does to different frequencies"""
    print("\n🎭 Analyzing mask behavior...")
    
    model.eval()
    all_masks = []
    freq_gains = []
    
    with torch.no_grad():
        for audio_path in tqdm(test_files[:10], desc="Processing"):
            wav, sr = torchaudio.load(audio_path)
            wav = convert_audio(wav, sr, SAMPLE_RATE, 1).to(device)
            
            stft = compute_stft(wav)
            mag = torch.abs(stft)
            model_input = preprocess_for_model(mag).transpose(1, 2)
            model_input = _ensure_batch_time_freq(model_input)
            mask = model(model_input)
            mask_np = mask.squeeze().detach().cpu().numpy()
            all_masks.append(mask_np)
            
            # Average gain per frequency bin
            freq_gains.append(mask_np.mean(axis=0))
    
    # --- FIX STARTS HERE ---
    # 1. Find the minimum time dimension across all masks
    min_time_dim = min(m.shape[0] for m in all_masks)
    
    # 2. Crop all masks to this minimum length
    all_masks_cropped = [m[:min_time_dim, :] for m in all_masks]
    
    # 3. Now compute the mean on the aligned arrays
    avg_mask = np.mean(all_masks_cropped, axis=0)
    # --- FIX ENDS HERE ---

    avg_freq_gain = np.mean(freq_gains, axis=0)
    
    # Plot 1: Mask heatmap
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    
    im1 = axes[0, 0].imshow(avg_mask.T, aspect='auto', origin='lower', 
                            cmap='RdYlGn', vmin=0, vmax=2)
    axes[0, 0].set_xlabel('Time Frame')
    axes[0, 0].set_ylabel('Frequency Bin')
    axes[0, 0].set_title('Average Mask (Green=Boost, Red=Suppress)')
    plt.colorbar(im1, ax=axes[0, 0], label='Mask Value')
    
    # Plot 2: Frequency response
    freqs = np.linspace(0, SAMPLE_RATE/2, len(avg_freq_gain))
    axes[0, 1].plot(freqs, avg_freq_gain, linewidth=2, color='darkblue')
    axes[0, 1].axhline(y=1.0, color='red', linestyle='--', label='Unity Gain')
    axes[0, 1].fill_between(freqs, 0, avg_freq_gain, alpha=0.3, color='skyblue')
    axes[0, 1].set_xlabel('Frequency (Hz)')
    axes[0, 1].set_ylabel('Average Gain')
    axes[0, 1].set_title('Frequency Response (Average Mask Gain)')
    axes[0, 1].set_xlim([0, 8000])
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].legend()
    
    # Plot 3: Mask value distribution
    # Use the CROPPED masks for the histogram to avoid shape errors if you flat-concat
    all_values = np.concatenate([m.flatten() for m in all_masks_cropped])
    
    axes[1, 0].hist(all_values, bins=100, alpha=0.7, color='teal', edgecolor='black')
    axes[1, 0].axvline(x=1.0, color='red', linestyle='--', linewidth=2, label='Unity')
    axes[1, 0].set_xlabel('Mask Value')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title(f'Mask Value Distribution (μ={all_values.mean():.3f}, σ={all_values.std():.3f})')
    axes[1, 0].set_xlim([0, 3])
    axes[1, 0].grid(alpha=0.3)
    axes[1, 0].legend()
    
    # Plot 4: Amplification vs Suppression zones
    suppression = (avg_mask < 0.9).sum() / avg_mask.size * 100
    unity = ((avg_mask >= 0.9) & (avg_mask <= 1.1)).sum() / avg_mask.size * 100
    amplification = (avg_mask > 1.1).sum() / avg_mask.size * 100
    
    zones = ['Suppression\n(<0.9)', 'Unity\n(0.9-1.1)', 'Amplification\n(>1.1)']
    values = [suppression, unity, amplification]
    colors = ['indianred', 'gold', 'limegreen']
    
    axes[1, 1].bar(zones, values, color=colors, edgecolor='black', linewidth=2)
    axes[1, 1].set_ylabel('Percentage (%)')
    axes[1, 1].set_title('Time-Frequency Zones')
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    for i, v in enumerate(values):
        axes[1, 1].text(i, v + 1, f'{v:.1f}%', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "plots" / "mask_behavior.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Mask behavior analysis complete")
    
    return avg_freq_gain
# ==============================================================================
# ANALYSIS 4: ABLATION STUDIES
# ==============================================================================

def run_ablation_studies(model, test_files, clean_dir):
    """Compare performance with different components disabled"""
    print("\n🔬 Running ablation studies...")
    
    model.eval()
    results = {
        'full_model': {'snr': [], 'si_sdr': [], 'spec_dist': []},
        'no_input_norm': {'snr': [], 'si_sdr': [], 'spec_dist': []},
        'no_skip_connections': {'snr': [], 'si_sdr': [], 'spec_dist': []},
        'linear_mask': {'snr': [], 'si_sdr': [], 'spec_dist': []},
    }
    
    with torch.no_grad():
        for audio_path in tqdm(test_files[:N_TEST_SAMPLES], desc="Ablation testing"):
            # Get clean reference
            file_id = audio_path.stem.replace('_unproc', '')
            clean_path = clean_dir / f"{file_id}.flac"
            
            if not clean_path.exists():
                continue
                
            noisy_wav, clean_wav = load_audio_pair(audio_path, clean_path)
            
            # Full model
            stft = compute_stft(noisy_wav)
            mag = torch.abs(stft)
            phase = torch.angle(stft)
            model_input = preprocess_for_model(mag).transpose(1, 2)
            model_input = _ensure_batch_time_freq(model_input)
            mask = model(model_input)
            enhanced_mag = mag * mask.transpose(1, 2).squeeze(0)
            enhanced_wav = stft_to_audio(torch.polar(enhanced_mag, phase))
            
            results['full_model']['snr'].append(compute_snr(clean_wav, enhanced_wav).item())
            results['full_model']['si_sdr'].append(compute_si_sdr(clean_wav, enhanced_wav).item())
            results['full_model']['spec_dist'].append(
                spectral_distance(compute_stft(clean_wav), compute_stft(enhanced_wav)).item()
            )
            
            # Ablation 1: No input normalization
            model.input_norm.eval()
            original_weight = model.input_norm.weight.data.clone()
            original_bias = model.input_norm.bias.data.clone()
            model.input_norm.weight.data.fill_(1.0)
            model.input_norm.bias.data.fill_(0.0)
            
            mask_no_norm = model(model_input)
            enhanced_mag_no_norm = mag * mask_no_norm.transpose(1, 2).squeeze(0)
            enhanced_wav_no_norm = stft_to_audio(torch.polar(enhanced_mag_no_norm, phase))
            
            results['no_input_norm']['snr'].append(compute_snr(clean_wav, enhanced_wav_no_norm).item())
            results['no_input_norm']['si_sdr'].append(compute_si_sdr(clean_wav, enhanced_wav_no_norm).item())
            results['no_input_norm']['spec_dist'].append(
                spectral_distance(compute_stft(clean_wav), compute_stft(enhanced_wav_no_norm)).item()
            )
            
            model.input_norm.weight.data = original_weight
            model.input_norm.bias.data = original_bias
            
            # Ablation 2: Linear mask (sigmoid instead of ReLU)
            mask_linear = torch.sigmoid(mask)
            enhanced_mag_linear = mag * mask_linear.transpose(1, 2).squeeze(0)
            enhanced_wav_linear = stft_to_audio(torch.polar(enhanced_mag_linear, phase))
            
            results['linear_mask']['snr'].append(compute_snr(clean_wav, enhanced_wav_linear).item())
            results['linear_mask']['si_sdr'].append(compute_si_sdr(clean_wav, enhanced_wav_linear).item())
            results['linear_mask']['spec_dist'].append(
                spectral_distance(compute_stft(clean_wav), compute_stft(enhanced_wav_linear)).item()
            )
    
    # Plot results
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    models = list(results.keys())
    metrics = ['snr', 'si_sdr', 'spec_dist']
    titles = ['SNR (dB) ↑', 'SI-SDR (dB) ↑', 'Spectral Distance ↓']
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        means = [np.mean(results[m][metric]) for m in models]
        stds = [np.std(results[m][metric]) for m in models]
        
        x = np.arange(len(models))
        bars = axes[idx].bar(x, means, yerr=stds, alpha=0.8, capsize=5, edgecolor='black')
        
        # Color code: best is green
        best_idx = np.argmax(means) if 'dist' not in metric else np.argmin(means)
        for i, bar in enumerate(bars):
            if i == best_idx:
                bar.set_color('limegreen')
            else:
                bar.set_color('lightcoral')
        
        axes[idx].set_xticks(x)
        axes[idx].set_xticklabels([m.replace('_', '\n') for m in models], rotation=0)
        axes[idx].set_ylabel(metric.upper())
        axes[idx].set_title(title)
        axes[idx].grid(axis='y', alpha=0.3)
        
        # Add value labels
        for i, (m, s) in enumerate(zip(means, stds)):
            axes[idx].text(i, m + s + 0.5, f'{m:.2f}', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "plots" / "ablation_results.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save numerical results
    with open(OUTPUT_DIR / "data" / "ablation_results.json", 'w') as f:
        json.dump({k: {metric: {'mean': float(np.mean(v[metric])), 
                               'std': float(np.std(v[metric]))}
                      for metric in metrics} 
                  for k, v in results.items()}, f, indent=2)
    
    print("✓ Ablation studies complete")

# ==============================================================================
# ANALYSIS 5: ATTENTION/SALIENCY MAPS
# ==============================================================================

def generate_attention_maps(model, test_file):
    """Generate gradient-based attention maps"""
    print("\n🔍 Generating attention maps...")
    
    model.eval()
    
    # Load audio
    wav, sr = torchaudio.load(test_file)
    wav = convert_audio(wav, sr, SAMPLE_RATE, 1).to(device)
    
    stft = compute_stft(wav)
    mag = torch.abs(stft)
    model_input = preprocess_for_model(mag).transpose(1, 2)
    model_input = _ensure_batch_time_freq(model_input)
    model_input.requires_grad = True
    
    # Forward pass
    mask = model(model_input)
    
    # Compute gradient of output w.r.t. input
    output_mean = mask.mean()
    output_mean.backward()
    
    # Gradient magnitude as "attention"
    attention = model_input.grad.abs().squeeze().cpu().numpy()
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Original spectrogram
    im1 = axes[0, 0].imshow(mag.squeeze().cpu().numpy().T, aspect='auto', 
                            origin='lower', cmap='magma')
    axes[0, 0].set_title('Input Magnitude Spectrogram')
    axes[0, 0].set_xlabel('Time Frame')
    axes[0, 0].set_ylabel('Frequency Bin')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Attention map
    im2 = axes[0, 1].imshow(attention.T, aspect='auto', origin='lower', cmap='hot')
    axes[0, 1].set_title('Attention Map (Gradient Magnitude)')
    axes[0, 1].set_xlabel('Time Frame')
    axes[0, 1].set_ylabel('Frequency Bin')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # Predicted mask
    im3 = axes[1, 0].imshow(mask.squeeze().detach().cpu().numpy().T, 
                            aspect='auto', origin='lower', cmap='RdYlGn', vmin=0, vmax=2)
    axes[1, 0].set_title('Predicted Mask')
    axes[1, 0].set_xlabel('Time Frame')
    axes[1, 0].set_ylabel('Frequency Bin')
    plt.colorbar(im3, ax=axes[1, 0])
    
    # Overlay attention on mask
    overlay = attention * mask.squeeze().detach().cpu().numpy()
    im4 = axes[1, 1].imshow(overlay.T, aspect='auto', origin='lower', cmap='viridis')
    axes[1, 1].set_title('Attention × Mask (Focus Regions)')
    axes[1, 1].set_xlabel('Time Frame')
    axes[1, 1].set_ylabel('Frequency Bin')
    plt.colorbar(im4, ax=axes[1, 1])
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "plots" / "attention_maps.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Attention maps complete")

# ==============================================================================
# ANALYSIS 6: PERFORMANCE ACROSS FREQUENCY BANDS
# ==============================================================================
def analyze_what_was_lost(model, test_files):
    print("\n🕵️ Analyzing the 'Lost' Signal...")
    model.eval()
    
    # Just take one file for visualization
    wav, sr = torchaudio.load(test_files[0])
    wav = convert_audio(wav, sr, SAMPLE_RATE, 1).to(device)
    stft = compute_stft(wav)
    mag = torch.abs(stft)
    phase = torch.angle(stft)
    
    # Get model output
    model_input = preprocess_for_model(mag).transpose(1, 2)
    model_input = _ensure_batch_time_freq(model_input)
    mask = model(model_input)
    
    # Reconstruct Enhanced
    enhanced_mag = mag * mask.transpose(1, 2).squeeze(0)
    
    # CALCULATE WHAT WAS REMOVED
    # (Original - Enhanced) shows what the model subtracted
    residual_mag = torch.abs(mag - enhanced_mag) 
    
    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 1. Original
    axes[0].imshow(torch.log(mag).cpu().numpy().squeeze(), aspect='auto', origin='lower', cmap='magma')
    axes[0].set_title("Original Noisy Input")
    
    # 2. Enhanced (Your Result)
    axes[1].imshow(torch.log(enhanced_mag + 1e-9).cpu().detach().numpy().squeeze(), aspect='auto', origin='lower', cmap='magma')
    axes[1].set_title("Your Model Output")
    
    # 3. The Difference (Crucial!)
    axes[2].imshow(torch.log(residual_mag + 1e-9).cpu().detach().numpy().squeeze(), aspect='auto', origin='lower', cmap='inferno')
    axes[2].set_title("What Your Model REMOVED")
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "plots" / "residual_analysis.png")
    print("✓ Check 'residual_analysis.png'. If you see bright lines in the 3rd plot, you deleted speech!")
def analyze_frequency_bands(model, test_files, clean_dir):
    """Analyze performance in different frequency bands"""
    print("\n🎵 Analyzing frequency band performance...")
    
    # Define frequency bands (Hz)
    bands = {
        'Sub-bass': (20, 60),
        'Bass': (60, 250),
        'Low-mid': (250, 500),
        'Mid': (500, 2000),
        'High-mid': (2000, 4000),
        'High': (4000, 8000)
    }
    
    model.eval()
    band_improvements = {band: [] for band in bands}
    
    with torch.no_grad():
        for audio_path in tqdm(test_files[:N_TEST_SAMPLES], desc="Band analysis"):
            file_id = audio_path.stem.replace('_unproc', '')
            clean_path = clean_dir / f"{file_id}.flac"
            
            if not clean_path.exists():
                continue
            
            noisy_wav, clean_wav = load_audio_pair(audio_path, clean_path)
            
            # Process
            stft_noisy = compute_stft(noisy_wav)
            stft_clean = compute_stft(clean_wav)
            
            mag = torch.abs(stft_noisy)
            phase = torch.angle(stft_noisy)
            model_input = preprocess_for_model(mag).transpose(1, 2)
            model_input = _ensure_batch_time_freq(model_input)
            mask = model(model_input)
            enhanced_mag = mag * mask.transpose(1, 2).squeeze(0)
            stft_enhanced = torch.polar(enhanced_mag, phase)
            
            # Calculate improvement per band
            freq_bins = torch.linspace(0, SAMPLE_RATE/2, mag.shape[1])
            
            for band_name, (f_low, f_high) in bands.items():
                mask_band = (freq_bins >= f_low) & (freq_bins < f_high)
                
                # SNR before and after
                noisy_band = stft_noisy[:, mask_band, :]
                clean_band = stft_clean[:, mask_band, :]
                enhanced_band = stft_enhanced[:, mask_band, :]
                
                snr_before = compute_snr(clean_band, noisy_band).item()
                snr_after = compute_snr(clean_band, enhanced_band).item()
                
                improvement = snr_after - snr_before
                band_improvements[band_name].append(improvement)
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    band_names = list(bands.keys())
    means = [np.mean(band_improvements[b]) for b in band_names]
    stds = [np.std(band_improvements[b]) for b in band_names]
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(band_names)))
    bars = ax.bar(range(len(band_names)), means, yerr=stds, color=colors, 
                   alpha=0.8, capsize=5, edgecolor='black', linewidth=2)
    
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2, label='No improvement')
    ax.set_xticks(range(len(band_names)))
    ax.set_xticklabels(band_names, rotation=45, ha='right')
    ax.set_ylabel('SNR Improvement (dB)')
    ax.set_title('Performance Across Frequency Bands')
    ax.grid(axis='y', alpha=0.3)
    ax.legend()
    
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 0.3, f'{m:.2f}', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "plots" / "frequency_band_performance.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Frequency band analysis complete")

# ==============================================================================
# MAIN RUNNER
# ==============================================================================

def main():
    print("\n" + "="*70)
    print("🚀 COMPREHENSIVE MODEL ABLATION & ANALYSIS")
    print("="*70)
    
    # Load model
    print(f"\n📦 Loading model from {CHECKPOINT_PATH}")
    model = ConvMaskingUNet(n_freq_bins=1025).to(device)
    
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"   Epoch: {checkpoint.get('epoch', 'unknown')}")
        print(f"   Loss: {checkpoint.get('loss', 'unknown'):.4f}")
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    # Get test files
    test_files = list(TEST_AUDIO_DIR.glob("*_unproc.flac"))[:N_TEST_SAMPLES]
    print(f"\n📁 Found {len(test_files)} test files")
    
    if len(test_files) == 0:
        print("❌ No test files found! Check TEST_AUDIO_DIR path.")
        return
    
    # Run all analyses
    try:
        analyze_weight_distributions(model)
        analyze_feature_activations(model, test_files)
        analyze_mask_behavior(model, test_files)
        run_ablation_studies(model, test_files, CLEAN_AUDIO_DIR)
        generate_attention_maps(model, test_files[0])
        analyze_frequency_bands(model, test_files, CLEAN_AUDIO_DIR)
        analyze_what_was_lost(model, test_files)
        
        print("\n" + "="*70)
        print("✅ ANALYSIS COMPLETE!")
        print(f"📊 Results saved to: {OUTPUT_DIR}")
        print("="*70)
        
        print("\n📈 Generated plots:")
        for plot in sorted((OUTPUT_DIR / "plots").glob("*.png")):
            print(f"   • {plot.name}")
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()