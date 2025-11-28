import os
import shutil
import subprocess
from pathlib import Path
import sys
import torch 
import torchaudio

# ==============================================================================
# CONFIGURATION
# ==============================================================================

ROOT_DIR = Path(".") 

# Enhanced Audio Paths
ENHANCED_VALID_DIR = Path("Enhanced_Output/valid")
ENHANCED_EVAL_DIR = Path("Enhanced_Output/eval")

# Outputs
SHADOW_ROOT = Path("Shadow_Dataset")
EXP_DIR = Path("exp") 

def find_file(name, root):
    """Recursively finds a file."""
    matches = list(root.rglob(name))
    if matches: return matches[0]
    return None

def main():
    print("--- AUTO-CONFIGURING PATHS (Force Stereo Fix) ---")
    
    # 1. FIND LIBRARY & SCRIPTS
    clarity_lib_path = None
    for p in list(ROOT_DIR.rglob("clarity")):
        if (p / "utils").exists():
            clarity_lib_path = p
            break
            
    if not clarity_lib_path:
        print("❌ CRITICAL: Could not find 'clarity' library.")
        return
    
    python_path_root = clarity_lib_path.parent.absolute()
    print(f"✓ Found Clarity: {clarity_lib_path}")

    compute_whisper_script = find_file("compute_whisper.py", ROOT_DIR)
    recipes_dir = compute_whisper_script.parent
    predict_script = recipes_dir / "predict.py"

    # 2. FIND SCORES
    train_score_file = find_file("cadenza_data.train.whisper.jsonl", ROOT_DIR)
    if not train_score_file:
        train_score_file = find_file("cadenza_data.train.whisper.mixture.jsonl", ROOT_DIR)

    EXP_DIR.mkdir(exist_ok=True)
    if train_score_file:
        shutil.copy(train_score_file, EXP_DIR / "cadenza_data.train.whisper.jsonl")
    else:
        print("❌ WARNING: Training scores file not found.")

    # 3. FIND METADATA
    valid_meta_file = find_file("valid_metadata.json", ROOT_DIR)
    meta_root = valid_meta_file.parent
    
    # ==========================================================================
    # EXECUTION
    # ==========================================================================
    
    for split in ["valid", "eval"]:
        print(f"\n--- Processing {split} set ---")
        
        src_audio = ENHANCED_VALID_DIR if split == "valid" else ENHANCED_EVAL_DIR
        if not src_audio.exists(): continue
            
        target_sig_dir = SHADOW_ROOT / "cadenza_data" / "audio" / split / "signals"
        # CLEANUP: Remove old folder to ensure no mono files remain
        if target_sig_dir.exists():
            print("Cleaning up old shadow files...")
            shutil.rmtree(target_sig_dir)
        target_sig_dir.mkdir(parents=True, exist_ok=True)
        
        target_meta_dir = SHADOW_ROOT / "cadenza_data" / "metadata"
        target_meta_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy Metadata
        src_meta = meta_root / f"{split}_metadata.json"
        if src_meta.exists(): shutil.copy(src_meta, target_meta_dir)

        # --- THE FIX: FORCE STEREO CONVERSION ---
        files = list(src_audio.glob("*.wav")) + list(src_audio.glob("*.flac"))
        print(f"Converting {len(files)} files to Stereo FLAC...")
        
        for f in files:
            song_id = f.stem.replace("_enhanced", "").replace("_unproc", "")
            dest_path = target_sig_dir / f"{song_id}.flac"
            
            # Load: [Channels, Time]
            wav, sr = torchaudio.load(f)
            
            # Force Stereo if Mono [1, T] -> [2, T]
            if wav.shape[0] == 1:
                wav = torch.cat([wav, wav], dim=0)
            
            # Save as FLAC (Overwrite!)
            torchaudio.save(dest_path, wav, sr)

        # ENV SETUP
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{python_path_root}:{env.get('PYTHONPATH', '')}"

        # RUN WHISPER
        print(f">>> Running compute_whisper.py...")
        cmd_whisper = [
            "python", str(compute_whisper_script),
            f"data.cadenza_data_root={SHADOW_ROOT.absolute()}",
            f"split={split}",
            "baseline=whisper",
            "baseline.system=whisper",
            "hydra.run.dir=.",
            "baseline.reference=processed"
        ]
        
        try:
            subprocess.run(cmd_whisper, check=True, env=env)
        except subprocess.CalledProcessError as e:
            print(f"Whisper failed: {e}")
            continue

        # RUN PREDICT
        print(f">>> Running predict.py...")
        cmd_predict = [
            "python", str(predict_script),
            f"data.cadenza_data_root={SHADOW_ROOT.absolute()}",
            f"split={split}",
            "baseline=whisper",
            "baseline.system=whisper",
            "hydra.run.dir=.",
            "baseline.reference=processed"
        ]
        
        try:
            subprocess.run(cmd_predict, check=True, env=env)
            print("✅ Step Complete.")
        except subprocess.CalledProcessError as e:
            print(f"Predict failed: {e}")

    # RESULTS
    print("\n" + "="*60)
    print("CHECKING FOR OUTPUTS")
    csvs = list(Path(".").glob("*.csv")) + list(EXP_DIR.glob("*.csv"))
    for c in csvs:
        print(f"Found CSV: {c}")

if __name__ == "__main__":
    main()