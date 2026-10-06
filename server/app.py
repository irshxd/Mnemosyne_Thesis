import cv2
import time
import sys
import os
import multiprocessing
import datetime
import subprocess

# Ensure server directory is in python path for process B
SERVER_DIR = os.path.dirname(os.path.abspath(__file__))
if SERVER_DIR not in sys.path:
    sys.path.insert(0, SERVER_DIR)

# --- CONFIGURATION ---
SCAN_INTERVAL = 0.5   

def log(msg):
    timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] {msg}", flush=True)

# --- PROCESS B: THE BRAIN (GPU/MPS/CUDA) ---
def brain_process_entry(input_queue, output_queue):
    try:
        if SERVER_DIR not in sys.path:
            sys.path.insert(0, SERVER_DIR)
        from vision_engine import MnemosyneAI
        
        log("⏳ [BRAIN] Loading VLM Model into memory...")
        ai = MnemosyneAI()
        log("✅ [BRAIN] Model Ready for Inference.")
        
        last_spoken = ""
        last_speak_time = 0

        while True:
            # 1. Wait for frame from camera process
            frame = input_queue.get()
            
            # 2. Analyze frame with VLM
            start = time.time()
            text = ai.describe_scene(frame, max_tokens=20)
            duration = time.time() - start
            
            # 3. Send result back to main process
            output_queue.put({'text': text, 'duration': duration})
            
            # 4. Speak / Log text output (Only speak when text changes or re-announce hazard after delay)
            now = time.time()
            is_hazard = any(w in text.lower() for w in ["hazard", "caution", "danger", "step", "stair", "stop", "obstacle"])
            
            if text and text != last_spoken:
                ai.speak(text)
                last_spoken = text
                last_speak_time = now
            elif text and is_hazard and (now - last_speak_time > 8.0):
                ai.speak(text)
                last_speak_time = now
                    
    except Exception as e:
        log(f"❌ [BRAIN] CRITICAL ERROR: {e}")

def get_macos_camera_names():
    names = {}
    if sys.platform == 'darwin':
        try:
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

def discover_available_cameras(mac_cams):
    discovered = {}
    for idx in range(4):
        for api_pref in ([cv2.CAP_AVFOUNDATION, cv2.CAP_ANY] if sys.platform == 'darwin' else [cv2.CAP_ANY]):
            try:
                c = cv2.VideoCapture(idx, api_pref)
                if c.isOpened():
                    for warmup in range(5):
                        ret, frame = c.read()
                        if ret and frame is not None and frame.size > 0:
                            name = mac_cams.get(idx, f"Camera {idx}")
                            discovered[idx] = name
                            break
                        time.sleep(0.03)
                    c.release()
                if idx in discovered: break
            except Exception: pass
    if not discovered:
        discovered[0] = "Default Camera 0"
    return discovered

def open_camera_stream(idx):
    for api_pref in ([cv2.CAP_AVFOUNDATION, cv2.CAP_ANY] if sys.platform == 'darwin' else [cv2.CAP_ANY]):
        try:
            c = cv2.VideoCapture(idx, api_pref)
            if c.isOpened():
                for warmup in range(10):
                    ret, frame = c.read()
                    if ret and frame is not None and frame.size > 0:
                        return c
                    time.sleep(0.1)
                c.release()
        except Exception: pass
    return None

# --- PROCESS A: THE BODY (Camera/UI/Terminal Output) ---
def run_system():
    mac_cams = get_macos_camera_names()
    log("🔍 Scanning available camera devices...")
    available_cams = discover_available_cameras(mac_cams)
    available_indices = sorted(list(available_cams.keys()))
    
    log("📸 Available Cameras Discovered:")
    for idx, name in available_cams.items():
        log(f"   • Index {idx}: {name}")

    # Check CLI or ENV overrides
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
        if 0 in available_cams and "OBS" in available_cams[0] and len(available_indices) > 1:
            preferred_cam = available_indices[1]
            log(f"ℹ️ Auto-bypassing OBS Virtual Camera at Index 0. Defaulting to physical camera Index {preferred_cam} [{available_cams[preferred_cam]}]")
        else:
            preferred_cam = available_indices[0]

    # 1. SETUP QUEUES
    input_queue = multiprocessing.Queue(maxsize=1)
    output_queue = multiprocessing.Queue()

    # 2. START BRAIN PROCESS
    log("🚀 Launching AI Brain Process...")
    p = multiprocessing.Process(target=brain_process_entry, args=(input_queue, output_queue))
    p.start()

    # 3. START INITIAL CAMERA
    connected_idx = preferred_cam
    log(f"🎥 Opening Active Camera Stream (Index {connected_idx}: {available_cams.get(connected_idx, 'Camera')})...")
    cap = open_camera_stream(connected_idx)

    if not cap:
        log("⚠️ Preferred camera failed, scanning remaining indices...")
        for fallback_idx in available_indices:
            cap = open_camera_stream(fallback_idx)
            if cap:
                connected_idx = fallback_idx
                break

    if not cap:
        log("❌ No Active Camera Device Found!")
        p.terminate()
        sys.exit(1)

    log(f"✅ Active Feed: Camera Index {connected_idx} [{available_cams.get(connected_idx, 'Camera')}]")

    last_scan_time = time.time()
    is_brain_busy = False 
    latest_text = "Analyzing scene..."
    has_display = os.environ.get('DISPLAY') is not None or sys.platform == 'darwin'
    
    toast_msg = f"ACTIVE: [{connected_idx}] {available_cams.get(connected_idx, 'Camera')[:20]}"
    toast_time = time.time() + 2.5

    log("\n=================================================================")
    log("👁️ MNEMOSYNE RUNNING: Point camera at your surroundings")
    log("🕹️ UI CONTROLS: Press 'c' to cycle cameras, [0-9] to switch, 'q' to quit.")
    log("=================================================================\n")

    failed_captures = 0

    try:
        while True:
            # --- CHECK FOR AI RESPONSES ---
            while not output_queue.empty():
                result = output_queue.get()
                latest_text = result['text']
                latest_duration = result['duration']
                log(f"⚡ [AI SCENE READOUT] ({latest_duration:.2f}s latency): \"{latest_text}\"")
                is_brain_busy = False

            # --- CAPTURE FRAME WITH AUTO-RECONNECT ---
            ret, frame = cap.read()
            if not ret or frame is None:
                failed_captures += 1
                if failed_captures >= 5:
                    log(f"⚠️ Stream lost on Camera {connected_idx}. Attempting reconnect...")
                    if cap:
                        try: cap.release()
                        except: pass
                    cap = open_camera_stream(connected_idx)
                    if cap:
                        log(f"✅ Reconnected to Camera Index {connected_idx}")
                        failed_captures = 0
                    else:
                        time.sleep(0.3)
                        continue
                time.sleep(0.05)
                continue
            else:
                failed_captures = 0

            # --- SEND FRAME TO BRAIN ---
            now = time.time()
            if (now - last_scan_time > SCAN_INTERVAL) and not is_brain_busy:
                h, w = frame.shape[:2]
                if w > 378:
                    scale = 378 / w
                    small_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
                else:
                    small_frame = frame

                if not input_queue.full():
                    input_queue.put(small_frame)
                    is_brain_busy = True
                    last_scan_time = now

            # --- DISPLAY WINDOW / FUTURISTIC HUD OVERLAY ---
            if has_display:
                try:
                    h, w = frame.shape[:2]
                    
                    # 1. TOP HEADER BANNER (Semi-transparent dark overlay)
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (0, 0), (w, 52), (15, 15, 15), -1)
                    
                    # 2. BOTTOM SUBTITLE PANEL
                    panel_height = 70
                    is_hazard = any(w in latest_text.lower() for w in ["hazard", "caution", "danger", "step", "stair", "stop", "obstacle"])
                    panel_color = (0, 0, 180) if is_hazard else (20, 20, 20)
                    cv2.rectangle(overlay, (0, h - panel_height), (w, h), panel_color, -1)
                    
                    # Blend overlay with original frame (alpha=0.85)
                    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

                    # 3. TOP BANNER TEXT & STATUS BADGE
                    cv2.putText(frame, "MNEMOSYNE EDGE VISION  |  4-BIT VLM (OFFLINE)", 
                                (15, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)
                    
                    # Camera Selector Badge (Top Header line 2)
                    cam_selector_str = "CAMS: " + "  ".join([f"[{i}]{'*' if i == connected_idx else ''} {available_cams[i][:12]}" for i in available_indices])
                    cv2.putText(frame, cam_selector_str[:65], 
                                (15, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 220, 255), 1, cv2.LINE_AA)

                    # Status Badge (Top Right)
                    status_str = "THINKING..." if is_brain_busy else "READY"
                    status_color = (0, 165, 255) if is_brain_busy else (0, 255, 120)
                    cv2.putText(frame, f"[{status_str}]", (w - 130, 24), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.50, status_color, 2, cv2.LINE_AA)

                    # 4. CENTER TARGET CORNER BRACKETS (Scanning Reticle)
                    box_w, box_h = int(w * 0.5), int(h * 0.5)
                    x1, y1 = int((w - box_w)/2), int((h - box_h)/2)
                    x2, y2 = x1 + box_w, y1 + box_h
                    line_len = 25
                    bracket_color = (0, 255, 255) if is_brain_busy else (100, 100, 100)
                    
                    # Reticle corners
                    cv2.line(frame, (x1, y1), (x1 + line_len, y1), bracket_color, 2)
                    cv2.line(frame, (x1, y1), (x1, y1 + line_len), bracket_color, 2)
                    cv2.line(frame, (x2, y1), (x2 - line_len, y1), bracket_color, 2)
                    cv2.line(frame, (x2, y1), (x2, y1 + line_len), bracket_color, 2)
                    cv2.line(frame, (x1, y2), (x1 + line_len, y2), bracket_color, 2)
                    cv2.line(frame, (x1, y2), (x1, y2 - line_len), bracket_color, 2)
                    cv2.line(frame, (x2, y2), (x2 - line_len, y2), bracket_color, 2)
                    cv2.line(frame, (x2, y2), (x2, y2 - line_len), bracket_color, 2)

                    # 5. CAMERA SWITCH TOAST NOTIFICATION (Center Banner)
                    if time.time() < toast_time:
                        toast_overlay = frame.copy()
                        cv2.rectangle(toast_overlay, (int(w*0.15), int(h*0.42)), (int(w*0.85), int(h*0.54)), (0, 100, 0), -1)
                        cv2.addWeighted(toast_overlay, 0.8, frame, 0.2, 0, frame)
                        cv2.putText(frame, f"🔄 {toast_msg}", (int(w*0.18), int(h*0.49)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)

                    # 6. BOTTOM READOUT CARD
                    prefix = "⚠️ HAZARD DETECTED: " if is_hazard else "👁️ SCENE: "
                    full_readout = prefix + latest_text
                    text_color = (0, 255, 255) if not is_hazard else (255, 255, 255)
                    
                    cv2.putText(frame, full_readout[:75], (15, h - 38), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, text_color, 2, cv2.LINE_AA)
                    
                    # Sub-legend with Key controls
                    latency_str = f"Latency: {latest_duration:.2f}s" if 'latest_duration' in locals() else "Scanning..."
                    cv2.putText(frame, f"{latency_str}  |  'r' Rescan  |  'c'/[0-9] Switch Cam  |  'q' Quit", 
                                (15, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

                    cv2.imshow("Mnemosyne Edge Vision HUD", frame)
                    
                    # --- KEYBOARD CONTROLS & LIVE CAMERA SWITCHING ---
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == ord('Q'):
                        break
                    elif key == ord('r') or key == ord('R'):
                        log("🔄 User pressed 'r': Rescanning connected hardware cameras...")
                        mac_cams = get_macos_camera_names()
                        available_cams = discover_available_cameras(mac_cams)
                        available_indices = sorted(list(available_cams.keys()))
                        toast_msg = f"RESCANNED: FOUND {len(available_cams)} CAMERA(S)"
                        toast_time = time.time() + 2.5
                        log(f"📸 Cameras found after rescan: {available_cams}")
                    elif key == ord('c') or key == ord('C'):
                        # Cycle to next camera
                        curr_pos = available_indices.index(connected_idx) if connected_idx in available_indices else 0
                        next_idx = available_indices[(curr_pos + 1) % len(available_indices)]
                        log(f"🔄 User pressed 'c': Switching camera from Index {connected_idx} to Index {next_idx} [{available_cams.get(next_idx, 'Camera')}]...")
                        if cap: cap.release()
                        new_cap = open_camera_stream(next_idx)
                        if new_cap:
                            cap = new_cap
                            connected_idx = next_idx
                            toast_msg = f"SWITCHED TO CAM [{connected_idx}]: {available_cams.get(connected_idx, 'Camera')[:20]}"
                            toast_time = time.time() + 2.5
                            log(f"✅ Camera Switched Successfully -> Index {connected_idx}")
                    elif ord('0') <= key <= ord('9'):
                        target_num = key - ord('0')
                        if target_num in available_indices and target_num != connected_idx:
                            log(f"🔄 User pressed '{target_num}': Switching camera to Index {target_num} [{available_cams.get(target_num, 'Camera')}]...")
                            if cap: cap.release()
                            new_cap = open_camera_stream(target_num)
                            if new_cap:
                                cap = new_cap
                                connected_idx = target_num
                                toast_msg = f"SWITCHED TO CAM [{connected_idx}]: {available_cams.get(connected_idx, 'Camera')[:20]}"
                                toast_time = time.time() + 2.5
                                log(f"✅ Camera Switched Successfully -> Index {connected_idx}")

                except Exception as e:
                    log(f"UI Error: {e}")
                    has_display = False

            time.sleep(0.01)

    except KeyboardInterrupt:
        log("\n🛑 Stopping Mnemosyne...")
    finally:
        log("🧹 Cleaning up system resources...")
        p.terminate()
        p.join()
        if cap:
            cap.release()
        if has_display:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        log("✅ Mnemosyne Shutdown Complete.")

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    run_system()