#!/usr/bin/env python3
"""
Project Mnemosyne - System Diagnostic Tool
Usage: ./venv/bin/python diagnose.py
"""

import os
import sys
import time
import cv2
import psutil
import numpy as np

# Ensure server module and project root are accessible
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SERVER_DIR = os.path.join(PROJECT_ROOT, "server")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if SERVER_DIR not in sys.path:
    sys.path.insert(0, SERVER_DIR)

def print_banner():
    print("=" * 65)
    print(" 👁️ PROJECT MNEMOSYNE - SYSTEM DIAGNOSTICS & HARDWARE CHECK")
    print("=" * 65)

def check_memory():
    print("\n[1. MEMORY & HARDWARE CHECK]")
    mem = psutil.virtual_memory()
    total_gb = mem.total / (1024**3)
    avail_gb = mem.available / (1024**3)
    print(f"   • Total RAM     : {total_gb:.2f} GB")
    print(f"   • Available RAM : {avail_gb:.2f} GB")
    if avail_gb < 1.5:
        print("   ⚠️  WARNING: Low available memory! Close background apps for best performance.")
    else:
        print("   ✅ Memory status: OPTIMAL for Edge VLM.")

def get_macos_camera_names():
    names = {}
    if sys.platform == 'darwin':
        try:
            import subprocess
            res = subprocess.run(['system_profiler', 'SPCameraDataType'], capture_output=True, text=True, timeout=3)
            idx = 0
            for line in res.stdout.splitlines():
                s = line.strip()
                if s.endswith(':') and not any(s.startswith(k) for k in ['Model ID', 'Unique ID', 'Camera', 'SPI', 'Control']):
                    names[idx] = s[:-1]
                    idx += 1
        except Exception:
            pass
    return names

def check_camera():
    print("\n[2. CAMERA DISCOVERY & SENSOR CHECK]")
    mac_cams = get_macos_camera_names()
    
    explicit_cam = None
    for i, arg in enumerate(sys.argv):
        if arg in ["--camera", "-c"] and i + 1 < len(sys.argv):
            try: explicit_cam = int(sys.argv[i + 1])
            except ValueError: pass
        elif arg.startswith("--camera="):
            try: explicit_cam = int(arg.split("=")[1])
            except ValueError: pass
    if explicit_cam is None and "CAMERA_INDEX" in os.environ:
        try: explicit_cam = int(os.environ["CAMERA_INDEX"])
        except ValueError: pass

    if explicit_cam is not None:
        preferred_cam = explicit_cam
    else:
        if 0 in mac_cams and "OBS" in mac_cams[0]:
            preferred_cam = 1
            print(f"   ℹ️ Detected 'OBS Virtual Camera' at Index 0. Auto-bypassing to physical camera at Index 1 [{mac_cams.get(1, 'Physical Camera')}]")
        else:
            preferred_cam = 0

    search_indices = [preferred_cam] + [i for i in [1, 0, 2] if i != preferred_cam]
    active_cap = None
    for idx in search_indices:
        for api_pref in ([cv2.CAP_AVFOUNDATION, cv2.CAP_ANY] if sys.platform == 'darwin' else [cv2.CAP_ANY]):
            try:
                c = cv2.VideoCapture(idx, api_pref)
                if c.isOpened():
                    for warmup in range(10):
                        ret, frame = c.read()
                        if ret and frame is not None and frame.size > 0:
                            h, w = frame.shape[:2]
                            cam_name = mac_cams.get(idx, f"Device {idx}")
                            print(f"   ✅ Camera Index {idx} [{cam_name}]: CONNECTED ({w}x{h} resolution)")
                            active_cap = c
                            break
                        time.sleep(0.1)
                    if active_cap:
                        break
                    c.release()
            except Exception:
                pass
        if active_cap:
            break
    
    if not active_cap:
        print("   ❌ Camera Error: No active camera device detected.")
        print("      Check macOS Privacy > Camera settings or USB webcam connection.")
        return False
    else:
        active_cap.release()
        return True

def check_vlm_engine():
    print("\n[3. VISION-LANGUAGE MODEL (VLM) DIAGNOSTIC]")
    local_4bit = os.path.join(PROJECT_ROOT, "moondream-4bit")
    if os.path.exists(local_4bit):
        print(f"   • Model Source   : Quantized 4-bit Local Model ({local_4bit})")
    else:
        print("   • Model Source   : Remote HuggingFace ('vikhyatk/moondream2')")

    try:
        try:
            from server.vision_engine import MnemosyneAI
        except ImportError:
            from vision_engine import MnemosyneAI
        
        print("   ⏳ Loading VLM into memory...")
        t0 = time.time()
        ai = MnemosyneAI()
        load_dt = time.time() - t0
        print(f"   ✅ VLM Loaded in {load_dt:.2f} seconds.")

        # Test inference pass
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(test_frame, "DIAGNOSTIC SCENE", (100, 240), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        print("   ⚡ Running Test Inference Pass...")
        t1 = time.time()
        readout = ai.describe_scene(test_frame, max_tokens=15)
        inf_dt = time.time() - t1
        
        print(f"   ✅ Test Inference Latency : {inf_dt:.2f} seconds")
        print(f"   💬 Model Readout Output  : \"{readout}\"")
        return True

    except Exception as e:
        print(f"   ❌ VLM Engine Error: {e}")
        return False

def main():
    print_banner()
    check_memory()
    cam_ok = check_camera()
    vlm_ok = check_vlm_engine()
    
    print("\n" + "=" * 65)
    if cam_ok and vlm_ok:
        print(" 🎉 SYSTEM DIAGNOSTIC PASSED: Mnemosyne is ready for live use!")
        print(" Run live visual server with: ./venv/bin/python server/app.py")
    else:
        print(" ⚠️  DIAGNOSTIC COMPLETED WITH WARNINGS/ERRORS. See details above.")
    print("=" * 65 + "\n")

if __name__ == "__main__":
    main()
