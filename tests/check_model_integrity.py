import os
import sys

# --- CONFIGURATION ---
# Auto-detect the path again
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ssd_cache_path = os.path.join(project_root, "data", "hf_cache")
base_model_dir = os.path.join(ssd_cache_path, "hub", "models--vikhyatk--moondream2", "snapshots")

print("=== 🔍 MODEL INTEGRITY CHECK ===")
print(f"Checking base dir: {base_model_dir}")

if not os.path.exists(base_model_dir):
    print("❌ ERROR: Base directory not found. Download failed completely.")
    sys.exit(1)

# Get the snapshot folder (the hash)
try:
    snapshots = os.listdir(base_model_dir)
    if not snapshots:
        print("❌ ERROR: 'snapshots' folder is empty.")
        sys.exit(1)
        
    snapshot_hash = snapshots[0]
    full_path = os.path.join(base_model_dir, snapshot_hash)
    print(f"📂 Snapshot found: {snapshot_hash}")
    print(f"   Path: {full_path}")
    
    # List all files
    print("\n--- 📄 FILES IN FOLDER ---")
    files = os.listdir(full_path)
    found_fourier = False
    
    for f in files:
        if f == "fourier_features.py":
            print(f"   ✅ {f} (CRITICAL FOUND)")
            found_fourier = True
        elif f.endswith(".py"):
            print(f"   🐍 {f}")
        elif f.endswith(".json"):
            print(f"   ⚙️ {f}")
        elif f.endswith(".safetensors") or f.endswith(".bin"):
            size_mb = os.path.getsize(os.path.join(full_path, f)) / (1024*1024)
            print(f"   🏋️ {f} ({size_mb:.2f} MB)")
            
    print("--------------------------")
    
    if found_fourier:
        print("\n✅ STATUS: File exists. The error is a Python Import Path issue.")
    else:
        print("\n❌ STATUS: 'fourier_features.py' is MISSING! The download was incomplete.")
        
except Exception as e:
    print(f"❌ Error reading folder: {e}")