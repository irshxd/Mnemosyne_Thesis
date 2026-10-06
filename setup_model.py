import subprocess
import sys

def build_quantized_model():
    print("🔨 Starting DIY Quantization (This takes ~2-3 mins)...")
    
    # 1. Update/Install to ensure we match the flags
    subprocess.run([sys.executable, "-m", "pip", "install", "-U", "mlx-vlm"], check=True)

    # 2. Run the conversion command with the NEW flags
    # --hf-path: The source model on HuggingFace
    # --mlx-path: Where to save it locally
    # -q: Enable quantization
    cmd = [
        sys.executable, "-m", "mlx_vlm.convert",
        "--hf-path", "vikhyatk/moondream2", 
        "--mlx-path", "moondream-4bit",
        "-q",
        "--q-group-size", "64",
        "--q-bits", "4"
    ]
    
    print(f"\n🚀 Running: {' '.join(cmd)}\n")
    try:
        subprocess.run(cmd, check=True)
        print("\n✅ SUCCESS: Custom 4-bit model saved to './moondream-4bit'")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Conversion Failed. Error code: {e.returncode}")

if __name__ == "__main__":
    build_quantized_model()