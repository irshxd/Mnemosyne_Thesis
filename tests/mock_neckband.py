import cv2
import time
import numpy as np
from flask import Flask, Response, jsonify
import math

# --- CONFIGURATION ---
# This mimics the ESP32's IP address and port
DEVICE_PORT = 5001 

app = Flask(__name__)
# Uses your Mac's Webcam as the "RGB Sensor"
camera = cv2.VideoCapture(1) 

# --- SIMULATION HELPERS ---
def generate_fake_thermal_data():
    """
    Simulates the MLX90640 Sensor (32x24 grid).
    Generates a background of ~25°C and a moving 'hot spot' of ~80°C.
    """
    # 1. Background noise (Room temp 25°C)
    thermal_grid = np.random.normal(25.0, 1.0, (24, 32))
    
    # 2. Create a moving hot spot (Simulating a cup or stove)
    t = time.time()
    # Math to make the spot move in a circle
    center_x = int(16 + 10 * math.sin(t))  
    center_y = int(12 + 8 * math.cos(t))   
    
    # 3. Apply heat to that spot (Radius of 4 pixels)
    for y in range(24):
        for x in range(32):
            dist = math.sqrt((x - center_x)**2 + (y - center_y)**2)
            if dist < 5:
                # Add heat: closer to center = hotter
                thermal_grid[y, x] += (80 - 25) * (1 - dist/5) 
    
    # Flatten to list (This is exactly how the ESP32 sends it)
    return thermal_grid.flatten().tolist()

def generate_video_frames():
    """Reads webcam and encodes as JPEG stream."""
    while True:
        success, frame = camera.read()
        if not success:
            break
        
        # Resize to 640x480 (Simulation of ESP32 resolution)
        frame = cv2.resize(frame, (640, 480))
        
        # Add a timestamp to prove it's live
        cv2.putText(frame, f"ESP32 SIMULATION: {time.time():.2f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        # Yield the frame in MJPEG format (Standard IP Camera format)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        time.sleep(0.05) # Simulate 20 FPS

# --- ENDPOINTS (The API) ---
@app.route('/video_feed')
def video_feed():
    # The Brain will connect here to see video
    return Response(generate_video_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/thermal')
def thermal_data():
    # The Brain will connect here to get temperature
    data = generate_fake_thermal_data()
    return jsonify({"temp_grid": data, "max_temp": max(data)})

if __name__ == '__main__':
    print(f"🔴 [MOCK DEVICE] ESP32 Simulator running at http://127.0.0.1:{DEVICE_PORT}")
    app.run(host='0.0.0.0', port=DEVICE_PORT, threaded=True)