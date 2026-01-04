import os
import io
import time
import logging
import asyncio
import argparse
import numpy as np
import piexif
from pathlib import Path
from logging.handlers import RotatingFileHandler
from dotenv import load_dotenv

# Third-party libraries
from PIL import Image
from tqdm.asyncio import tqdm  # For the progress bar
from google import genai
from google.genai import types

# ================= COLOR CODES (ANSI) =================
# This class provides colored terminal output so errors (Red) 
# and warnings (Yellow) are immediately visible.
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

# ==============================================================================
#                          CONFIGURATION & SETTINGS
# ==============================================================================

# --- PATHS ---
# The script will search for images recursively in this folder
DEFAULT_INPUT_FOLDER = "./input_images"
# The restored images will be saved here, maintaining the folder structure
DEFAULT_OUTPUT_FOLDER = "./restored_images"

# --- RESTORATION LOGIC ---
# Mode: "AUTO" (detects BW/Color), "RESTORE_COLOR" (force color), "RESTORE_BW" (force BW)
DEFAULT_MODE = "AUTO" 
DEFAULT_MODEL_ID = "gemini-3-pro-image-preview" 
DEFAULT_RESOLUTION = "4K" # Forces the API to use high resolution

# --- UPLOAD STRATEGY (PNG vs. JPEG) ---
# The script starts with PNG (lossless, best quality).
# If errors occur (Timeouts/503), it automatically switches to compact JPEG to stabilize the upload.
UPLOAD_FORMAT_PRIMARY = "PNG"       
UPLOAD_FORMAT_FALLBACK = "JPEG"     
UPLOAD_JPEG_QUALITY = 95            # Quality for the fallback upload
RETRIES_BEFORE_FALLBACK = 2         # After how many errors should we switch formats?

# --- TECHNICAL SETTINGS ---
PARALLEL_LIMIT = 4        # How many images to process simultaneously
DEFAULT_START_DELAY = 1.0 # Pause between starts (1 second is enough for multi-key)
MAX_RETRIES_PER_KEY = 3   # How often can ONE key fail (excluding Quota errors) before we give up?
DEFAULT_TIMEOUT = 600     # Time in seconds before Python cancels the task (10 min)
BW_THRESHOLD = 3.0        # Saturation threshold (0-100): Below this is considered Black & White
SKIP_EXISTING = True      # Standard behavior: Skip if file already exists
FILENAME_PREFIX = ""      # Optional: Text prefix for filenames

# --- SAVE SETTINGS (DOWNLOAD) ---
SAVE_FORMAT = "PNG"       # Format for local storage (Result)
SAVE_JPEG_QUALITY = 95    # Only relevant if SAVE_FORMAT is set to "JPEG"

SUFFIX_MAP = {
    "RESTORE_COLOR": "_restored",
    "RESTORE_BW": "_restored_bw",
    "RESTORE_BW_COLORIZE": "_colorized",
    "COLOR_TO_BW": "_bw_converted"
}

# ================= PROMPTS =================
# The detailed instructions sent to the AI model
PROMPT_BASE = (
    "You are an expert in professional photo restoration. Your task is to "
    "visually analyze the attached image and digitally restore it. "
    "1. Remove defects: Scratches, dust, tears, fungal growth, and chemical stains. "
    "2. Correct image errors: Remove color casts, moderately improve contrast and sharpness. "
    "3. Remove borders: Crop scanner frames, slide mounts, or film strips, or fill them "
    "   content-aware (inpainting) so they blend seamlessly with the image. "
    "4. IMPORTANT: Preserve the identity of people and details. Do not hallucinate new objects."
)

PROMPTS = {
    "RESTORE_COLOR": PROMPT_BASE + " The image is color. Restore natural, vibrant colors. Pay special attention to realistic skin tones.",
    "RESTORE_BW": PROMPT_BASE + " The image is black and white. Maintain the black and white aesthetic. Do NOT colorize it. Optimize tonal values for deep blacks and clear highlights.",
    "RESTORE_BW_COLORIZE": PROMPT_BASE + " The image is black and white. Restore it AND subsequently colorize it realistically. Choose historically plausible colors for clothing, environment, and skin tones.",
    "COLOR_TO_BW": PROMPT_BASE + " The image is color. Convert it into an aesthetic black and white photograph. Simulate the look of classic analog black and white film (e.g., Kodak Tri-X)."
}

# ================= LOGGING SETUP =================
# Writes logs to a file. Rotates file if it exceeds 5MB.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        RotatingFileHandler("restoration_log.txt", maxBytes=5*1024*1024, backupCount=3, encoding='utf-8'),
    ]
)
logging.getLogger("google").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)


# ================= INTELLIGENT KEY MANAGER =================
class KeyManager:
    """
    Manages multiple API keys.
    - Round Robin: Cycles through keys for load balancing.
    - Quota Protection: If a key reports '429 Quota Exceeded', 
      it is immediately removed from the active pool.
    """
    def __init__(self, api_keys):
        # We store the client and a masked version of the key for logging
        self.clients = [{"client": genai.Client(api_key=k), "key_mask": k[:4]+"..."} for k in api_keys]
        self.total_keys = len(api_keys)
        self.current_index = 0
        self.lock = asyncio.Lock() # Crucial for async safety

    async def get_client(self):
        """Returns the next available client (Round Robin)."""
        async with self.lock:
            if not self.clients:
                raise RuntimeError("ALL_KEYS_EXHAUSTED") # Signal: We are completely out of keys
            
            # Select next client
            client_data = self.clients[self.current_index % len(self.clients)]
            self.current_index += 1
            return client_data

    async def ban_current_key(self, client_obj):
        """Removes a client from the pool if it hits the quota limit."""
        async with self.lock:
            for i, c_data in enumerate(self.clients):
                if c_data["client"] == client_obj:
                    k_mask = c_data["key_mask"]
                    tqdm.write(f"{Colors.RED}💀 Key {k_mask} is depleted (Quota). Removing it!{Colors.RESET}")
                    logging.warning(f"Key {k_mask} removed due to Quota limit.")
                    self.clients.pop(i)
                    return

# ================= HELPER FUNCTIONS =================

def load_api_keys():
    """Loads MULTIPLE API keys from .env file (comma separated)."""
    load_dotenv()
    keys_str = os.getenv("GOOGLE_API_KEY")
    if not keys_str: raise ValueError("NO API KEY FOUND! Please check .env file.")
    
    # Split by comma, strip whitespace
    keys = [k.strip() for k in keys_str.split(',') if k.strip()]
    if not keys: raise ValueError("NO VALID API KEYS FOUND!")
    return keys

def is_image_bw(pil_img, threshold=BW_THRESHOLD):
    """Checks if an image is Black & White based on saturation."""
    img_small = pil_img.resize((100, 100)).convert("HSV")
    score = (np.mean(np.array(img_small)[:, :, 1]) / 255.0) * 100
    return score < threshold

def sanitize_exif(original_path, new_size):
    """Copies EXIF data, removes thumbnails, updates dimensions."""
    try:
        exif_dict = piexif.load(original_path)
        if "thumbnail" in exif_dict: del exif_dict["thumbnail"]
        if "0th" in exif_dict:
            exif_dict["0th"][piexif.ImageIFD.ImageWidth] = new_size[0]
            exif_dict["0th"][piexif.ImageIFD.ImageLength] = new_size[1]
        if "Exif" in exif_dict:
            exif_dict["Exif"][piexif.ImageIFD.PixelXDimension] = new_size[0]
            exif_dict["Exif"][piexif.ImageIFD.PixelYDimension] = new_size[1]
        return piexif.dump(exif_dict)
    except Exception: return None

# ================= API CALL (Basis) =================

async def try_generate(client, image_bytes, mime_type, prompt, resolution, timeout):
    """Executes ONE attempt with ONE specific client."""
    config = types.GenerateContentConfig(
        temperature=0.3,
        response_modalities=["IMAGE"],
        image_config=types.ImageConfig(image_size=resolution),
    )
    
    api_call = client.aio.models.generate_content(
        model=DEFAULT_MODEL_ID,
        contents=[
            types.Content(parts=[
                types.Part.from_text(text=prompt),
                types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
            ])
        ],
        config=config
    )
    return await asyncio.wait_for(api_call, timeout=timeout)

# ================= MAIN LOGIC PER IMAGE =================

async def process_image_smart(sem, key_manager, file_path, output_dir, input_base_path, mode, args, stats):
    """
    Manages the full lifecycle of an image:
    - Fetches a key.
    - If key is empty -> bans key -> fetches new key -> continues.
    - If upload fails -> switches from PNG to JPEG.
    - Replicates folder structure from input to output.
    """
    filename = file_path.name
    async with sem:
        try:
            with Image.open(file_path) as img:
                # Determine Mode
                is_bw = False
                current_mode = mode
                if mode == "AUTO":
                    is_bw = is_image_bw(img)
                    current_mode = "RESTORE_BW" if is_bw else "RESTORE_COLOR"
                
                jobs = [current_mode]
                if args.auto_colorize_bw and ((mode == "AUTO" and is_bw) or (mode == "RESTORE_BW")):
                    jobs.append("RESTORE_BW_COLORIZE")
                    tqdm.write(f"{Colors.CYAN}✨ Dual-Mode: {filename}{Colors.RESET}")

                for job_mode in jobs:
                    # --- RECURSIVE FOLDER MIRRORING ---
                    # Calculate relative path (e.g. "Vacation/1990/img.jpg")
                    relative_path = file_path.relative_to(input_base_path)
                    # Create target folder structure (e.g. "Output/Vacation/1990")
                    target_folder = output_dir / relative_path.parent
                    target_folder.mkdir(parents=True, exist_ok=True)
                    
                    suffix = SUFFIX_MAP.get(job_mode, "-processed")
                    out_filename = f"{args.prefix}{file_path.stem}{suffix}.{args.format.lower()}"
                    out_path = target_folder / out_filename
                    
                    # Smart Skip
                    if args.skip_existing and out_path.exists() and not args.force:
                        if abs(file_path.stat().st_mtime - out_path.stat().st_mtime) < 1.0:
                            stats["skip"] += 1
                            continue
                    
                    # === RETRY LOOP (Across multiple keys) ===
                    success = False
                    attempt_counter = 0 # Counts only "real" technical errors, not quota errors
                    
                    while not success:
                        try:
                            # 1. Get fresh key
                            client_data = await key_manager.get_client()
                            client = client_data["client"]
                            
                            # 2. Choose format (Smart Upload: PNG -> JPEG Fallback)
                            img_byte_arr = io.BytesIO()
                            use_fallback = attempt_counter >= RETRIES_BEFORE_FALLBACK
                            
                            if not use_fallback:
                                img.save(img_byte_arr, format=UPLOAD_FORMAT_PRIMARY)
                                mime_type = f"image/{UPLOAD_FORMAT_PRIMARY.lower()}"
                            else:
                                # Convert to RGB (essential for JPEG)
                                img.convert("RGB").save(img_byte_arr, format=UPLOAD_FORMAT_FALLBACK, quality=UPLOAD_JPEG_QUALITY)
                                mime_type = f"image/{UPLOAD_FORMAT_FALLBACK.lower()}"
                                if attempt_counter == RETRIES_BEFORE_FALLBACK:
                                    tqdm.write(f"{Colors.BLUE}Info: {filename} switching to JPEG (stability){Colors.RESET}")

                            # 3. Attempt API Call
                            response = await try_generate(
                                client, img_byte_arr.getvalue(), mime_type, 
                                PROMPTS[job_mode], args.resolution, args.timeout
                            )

                            # 4. Check Response & Save
                            if response and response.candidates:
                                candidate = response.candidates[0]
                                if hasattr(candidate, 'content') and candidate.content and candidate.content.parts:
                                    for part in candidate.content.parts:
                                        if part.inline_data:
                                            # Decode image
                                            restored_img = Image.open(io.BytesIO(part.inline_data.data))
                                            exif_bytes = sanitize_exif(str(file_path), restored_img.size)
                                            
                                            # Save settings
                                            save_kwargs = {}
                                            if args.format.upper() in ["JPG", "JPEG"]:
                                                save_kwargs["quality"] = args.quality
                                                save_kwargs["subsampling"] = 0 
                                                if restored_img.mode in ("RGBA", "P"): 
                                                    restored_img = restored_img.convert("RGB")
                                            if exif_bytes: save_kwargs["exif"] = exif_bytes

                                            # Save file
                                            restored_img.save(out_path, format=args.format, **save_kwargs)
                                            os.utime(out_path, (os.stat(file_path).st_atime, os.stat(file_path).st_mtime))
                                            
                                            stats["success"] += 1
                                            success = True
                                            break # Exit parts loop
                                    
                                    if success: break # Exit retry loop
                                    
                                    # Case: Parts exist but empty -> Safety Filter
                                    msg = f"No image data/Safety Filter for {filename}"
                                    logging.warning(msg)
                                    stats["error"] += 1 
                                    stats["failed_files"].append(f"{filename} (Blocked/Empty)")
                                    success = True # Abort, retries are useless for safety blocks
                                else:
                                    raise Exception("Empty Candidate Response")
                            else:
                                raise Exception("Empty API Response")

                        except Exception as e:
                            error_msg = str(e)
                            
                            # === CRITICAL: QUOTA HANDLING (429) ===
                            if "429" in error_msg:
                                # Key is dead -> Ban it and retry IMMEDIATELY with new key
                                await key_manager.ban_current_key(client)
                                continue # 'continue' restarts the while loop -> gets new key
                            
                            # === FATAL: ALL KEYS GONE ===
                            elif "ALL_KEYS_EXHAUSTED" in error_msg:
                                tqdm.write(f"{Colors.RED}❌ NO KEYS LEFT! Aborting: {filename}{Colors.RESET}")
                                stats["error"] += 1
                                stats["failed_files"].append(f"{filename} (No Keys left)")
                                return # Give up completely

                            # === NORMAL ERROR (503, Timeout, Network) ===
                            else:
                                attempt_counter += 1
                                wait_time = 2 ** attempt_counter
                                
                                if attempt_counter > MAX_RETRIES_PER_KEY:
                                    tqdm.write(f"{Colors.RED}❌ {filename}: Too many errors. Giving up.{Colors.RESET}")
                                    stats["error"] += 1
                                    stats["failed_files"].append(f"{filename} (Tech Error)")
                                    return
                                else:
                                    msg = "Server Error 503" if ("503" in error_msg or "Deadline" in error_msg) else str(e)
                                    tqdm.write(f"{Colors.YELLOW}⚠️  {filename}: {msg}. Waiting {wait_time}s.{Colors.RESET}")
                                    await asyncio.sleep(wait_time)

        except Exception as e:
            msg = f"System error at {filename}: {e}"
            logging.error(msg)
            tqdm.write(f"{Colors.RED}❌ {msg}{Colors.RESET}")
            stats["error"] += 1
            stats["failed_files"].append(f"{filename} (System Error)")

async def main():
    parser = argparse.ArgumentParser(description="AI Photo Restoration V8 (Recursive Mirror)")
    parser.add_argument("--input", default=DEFAULT_INPUT_FOLDER, help="Input folder")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_FOLDER, help="Output folder")
    parser.add_argument("--mode", default=DEFAULT_MODE, help="Mode")
    parser.add_argument("--parallel", type=int, default=PARALLEL_LIMIT, help="Threads")
    parser.add_argument("--format", default=SAVE_FORMAT, help="Save format")
    parser.add_argument("--quality", type=int, default=SAVE_JPEG_QUALITY, help="JPEG Quality")
    parser.add_argument("--skip_existing", type=bool, default=SKIP_EXISTING, help="Skip existing")
    parser.add_argument("--resolution", default=DEFAULT_RESOLUTION, help="Resolution")
    parser.add_argument("--start_delay", type=float, default=DEFAULT_START_DELAY, help="Delay")
    parser.add_argument("--prefix", default=FILENAME_PREFIX, help="Prefix")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="Timeout")
    parser.add_argument("--auto_colorize_bw", action="store_true", help="Dual-BW")
    parser.add_argument("--force", action="store_true", help="Force")
    
    args = parser.parse_args()
    if not args.auto_colorize_bw: args.auto_colorize_bw = True

    try:
        api_keys = load_api_keys()
        key_manager = KeyManager(api_keys)
        print(f"{Colors.HEADER}--- API SETUP (Smart Hybrid V8) ---{Colors.RESET}")
        print(f"Found Keys: {len(api_keys)}")
        print(f"Load Balancing: {Colors.GREEN}Active{Colors.RESET}")
        print(f"Folder Structure: {Colors.GREEN}Recursive Mirroring{Colors.RESET}")
    except ValueError as e:
        print(f"{Colors