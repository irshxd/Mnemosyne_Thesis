import cv2
import requests
import time
import psutil
import os
import sys

# Ensure server module can be imported
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SERVER_DIR = os.path.join(PROJECT_ROOT, "server")
if SERVER_DIR not in sys.path:
    sys.path.insert(0, SERVER_DIR)

NECKBAND_IP = "http://127.0.0.1:5001"

def print_status(step, msg, status="..."):
    print(f"[{step}] {msg:<50} {status}")

def check_memory():
    mem = psutil.virtual_memory()
    total_gb = mem.total / (1024**3)
    avail_gb = mem.available / (1024**3)
    print_status("1. MEMORY", f"Total RAM: {total_gb:.2f} GB | Available RAM: {avail_gb:.2f} GB")
    if mem.available < 2 * (1024**3):
        print("   ⚠️ WARNING: Low Available RAM! System might drop frame rates.")
    else:
        print("   ✅ Sufficient Memory for Edge VLM.")

def check_camera():
    print_status("2. CAMERA", "Testing Camera Connection...")
    cap = None
    for idx in [0, 1, 2]:
        try:
            c = cv2.VideoCapture(idx)
            if c.isOpened():
                ret, frame = c.read()
                if ret and frame is not None:
                    cap = c
                    print(f"   ✅ Active Camera Device {idx} Detected! Frame: {frame.shape[1]}x{frame.shape[0]}")
                    break
                c.release()
        except Exception:
            pass
    if not cap:
        print("   ⚠️ Local camera not detected. Checking mock network stream...")
        check_network()

def check_network():
    print_status("3. NETWORK MOCK", "Pinging ESP32 Neckband Simulator...")
    try:
        response = requests.get(f"{NECKBAND_IP}/thermal", timeout=2)
        if response.status_code == 200:
            print("   ✅ Mock Neckband is ALIVE and responding on Port 5001.")
        else:
            print(f"   ❌ Neckband Error: Status {response.status_code}")
    except requests.exceptions.ConnectionError:
        print("   ℹ️ Mock Neckband is offline (Normal if running webcam-only mode).")

def check_ai_load():
    print_status("4. AI ENGINE", "Loading MnemosyneAI VLM...")
    try:
        from vision_engine import MnemosyneAI
        start_time = time.time()
        ai = MnemosyneAI()
        elapsed = time.time() - start_time
        print(f"   ✅ AI Model Loaded Successfully in {elapsed:.2f} seconds.")
        
        # Test inference with dummy frame
        import numpy as np
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(dummy, "TEST", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            
        t0 = time.time()
        res = ai.describe_scene(dummy, max_tokens=15)
        dt = time.time() - t0
        print(f"   ⚡ Test Inference Latency: {dt:.2f}s | Result: \"{res}\"")

    except Exception as e:
        print(f"   ❌ AI LOAD FAILED: {e}")

if __name__ == "__main__":
    print("=== MNEMOSYNE EDGE DIAGNOSTICS ===")
    check_memory()
    print("-" * 60)
    check_camera()
    print("-" * 60)
    check_ai_load()
    print("-" * 60)
    print("Diagnostics Complete.")