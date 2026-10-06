import os
import sys
import subprocess
import cv2
import numpy as np
from PIL import Image

class MnemosyneAI:
    def __init__(self, model_path=None):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        local_4bit_path = os.path.join(project_root, "moondream-4bit")
        
        if model_path:
            self.model_path = model_path
        elif os.path.exists(local_4bit_path):
            self.model_path = local_4bit_path
            print(f"   ⚡ Using Local Quantized 4-bit Model: {local_4bit_path}")
        else:
            self.model_path = "vikhyatk/moondream2"
            print(f"   ⚡ Using HuggingFace Model: {self.model_path}")

        # Check for Apple Silicon / MLX support vs fallback
        self.use_mlx = False
        try:
            from mlx_vlm import load, generate
            self.mlx_load = load
            self.mlx_generate = generate
            self.use_mlx = True
            print("   ⚡ Hardware Engine: APPLE SILICON (MLX NATIVE)")
        except ImportError:
            print("   ⚠️ MLX VLM not available. Falling back to PyTorch/Transformers...")

        if self.use_mlx:
            try:
                self.model, self.processor = self.mlx_load(
                    self.model_path, 
                    trust_remote_code=True
                )
                print("   ✅ Model Loaded Successfully (MLX)")
            except Exception as e:
                print(f"❌ MLX Load Error: {e}")
                raise e
        else:
            # Fallback for Linux / Jetson / standard PyTorch
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
            print(f"   ⚡ Hardware Engine: PyTorch ({self.device.upper()})")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device != "cpu" else torch.float32
            ).to(self.device)
            print(f"   ✅ Model Loaded Successfully (PyTorch on {self.device})")

        # Automatic GPU Warmup Pass to eliminate initial 4-5s delay on frame 1
        try:
            dummy_frame = np.zeros((256, 256, 3), dtype=np.uint8)
            _ = self.describe_scene(dummy_frame, max_tokens=1)
            print("   🔥 Warmup Complete (Shaders & Weights Cached in GPU)")
        except Exception as e:
            print(f"   ⚠️ Warmup notice: {e}")

    def speak(self, text):
        """Cross-platform text-to-speech (audio output)."""
        try:
            # Terminate or skip if previous speech process is still active to avoid duplicate voice overlapping
            if hasattr(self, 'current_speech_proc') and self.current_speech_proc is not None:
                if self.current_speech_proc.poll() is None:
                    # Still speaking, skip to avoid double speech playback
                    return
            if sys.platform == "darwin":
                # macOS native TTS (non-blocking)
                self.current_speech_proc = subprocess.Popen(["say", "-r", "220", text])
            elif sys.platform.startswith("linux"):
                # Linux / Jetson / Pi TTS fallback using espeak
                self.current_speech_proc = subprocess.Popen(["espeak", "-s", "180", text], stderr=subprocess.DEVNULL)
        except Exception:
            pass # Graceful fallback if audio device unavailable

    def describe_scene(self, cv2_frame, max_tokens=20, prompt="Describe immediate path, obstacles, or hazards in under 10 words."):
        # 1. Resize (378px / 256px resolution cap for fast vision encoding)
        height, width = cv2_frame.shape[:2]
        if width > 378: 
            scale = 378 / width
            cv2_frame = cv2.resize(cv2_frame, (0,0), fx=scale, fy=scale)

        # 2. Convert to PIL
        rgb_frame = cv2.cvtColor(cv2_frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        
        # 3. Generate description
        if self.use_mlx:
            output = self.mlx_generate(
                self.model, 
                self.processor, 
                image=pil_image, 
                prompt=prompt,
                max_tokens=max_tokens,  
                verbose=False
            )
            # Handle string vs GenerationResult object vs dictionary output formats
            if hasattr(output, "text"):
                return output.text.strip()
            elif isinstance(output, str):
                return output.strip()
            elif isinstance(output, dict) and "text" in output:
                return output["text"].strip()
            return str(output).strip()
        else:
            enc_image = self.model.encode_image(pil_image)
            output = self.model.answer_question(enc_image, prompt, self.tokenizer, max_new_tokens=max_tokens)
            return output.strip()