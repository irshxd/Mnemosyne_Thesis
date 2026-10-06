import os
import sys
import time
import cv2
import numpy as np

# Ensure server module can be imported
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SERVER_DIR = os.path.join(PROJECT_ROOT, "server")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if SERVER_DIR not in sys.path:
    sys.path.insert(0, SERVER_DIR)

try:
    from server.vision_engine import MnemosyneAI
except ImportError:
    from vision_engine import MnemosyneAI

def run_latency_benchmark():
    print("=" * 60)
    print("⚡ MNEMOSYNE LATENCY BENCHMARK TOOL")
    print("=" * 60)

    # 1. Initialize AI Model
    print("⏳ Loading Model...")
    start_load = time.time()
    ai = MnemosyneAI()
    load_time = time.time() - start_load
    print(f"✅ Model Loaded in {load_time:.2f} seconds.\n")

    # 2. Create Synthetic Frame (mimicking camera resolution)
    # A colored test scene (e.g. simulated room frame)
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(test_frame, "BENCHMARK TEST FRAME", (50, 240), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.rectangle(test_frame, (100, 100), (300, 300), (0, 0, 255), -1) # Mock obstacle box

    # 3. Test Warmup Run
    print("🔥 Running Warmup Run...")
    _ = ai.describe_scene(test_frame, max_tokens=15)
    print("✅ Warmup Complete.\n")

    # 4. Benchmark Across Token Length Limits
    token_limits = [10, 20, 35]
    print("--- BENCHMARK RESULTS ---")
    print(f"{'Max Tokens':<12} | {'Avg Latency (s)':<18} | {'Sample Output'}")
    print("-" * 60)

    for limit in token_limits:
        times = []
        sample_output = ""
        for i in range(3):
            t0 = time.time()
            sample_output = ai.describe_scene(test_frame, max_tokens=limit)
            elapsed = time.time() - t0
            times.append(elapsed)
        
        avg_latency = sum(times) / len(times)
        print(f"{limit:<12} | {avg_latency:<18.3f} | '{sample_output}'")

    print("-" * 60)
    print("✅ Benchmark Finished.")

if __name__ == "__main__":
    run_latency_benchmark()
