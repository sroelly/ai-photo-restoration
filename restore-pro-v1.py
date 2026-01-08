import os
import io
import time
import logging
import asyncio
import argparse
import gc
import shutil
import numpy as np
import piexif
from datetime import datetime, timedelta
from pathlib import Path
from logging.handlers import RotatingFileHandler
from dotenv import load_dotenv

# Third-party libraries
from PIL import Image
from tqdm.asyncio import tqdm
from google import genai
from google.genai import types

# ================= COLOR CODES =================
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

# ================= SETTINGS =================

# --- PATHS ---
DEFAULT_INPUT_FOLDER = "./input_images"
DEFAULT_OUTPUT_FOLDER = "./restored_images"

# --- SYSTEM ---
MIN_DISK_SPACE_MB = 500   

# --- RESTORATION LOGIC ---
DEFAULT_MODE = "AUTO" 
DEFAULT_MODEL_ID = "gemini-3-pro-image-preview" 
DEFAULT_RESOLUTION = "4K" 

# --- UPLOAD STRATEGY ---
UPLOAD_FORMAT_PRIMARY = "PNG"       
UPLOAD_FORMAT_FALLBACK = "JPEG"     
UPLOAD_JPEG_QUALITY = 95            
RETRIES_BEFORE_FALLBACK = 1  

# --- TECHNICAL SETTINGS ---
PARALLEL_LIMIT = 2        
DEFAULT_START_DELAY = 0   
MAX_RETRIES_PER_KEY = 3   
DEFAULT_TIMEOUT = 1200    
BW_THRESHOLD = 3.0        
SKIP_EXISTING = True      
FILENAME_PREFIX = ""      

# --- SAVE SETTINGS ---
SAVE_FORMAT = "PNG"       
SAVE_JPEG_QUALITY = 95    

SUFFIX_MAP = {
    "RESTORE_COLOR": "_restored",
    "RESTORE_BW": "_restored_bw",
    "RESTORE_BW_COLORIZE": "_colorized",
    "COLOR_TO_BW": "_bw_converted"
}

# ================= PROMPTS =================
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
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        RotatingFileHandler("restoration_log.txt", maxBytes=5*1024*1024, backupCount=3, encoding='utf-8'),
    ]
)
logging.getLogger("google").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

# ================= HELPER FUNCTIONS =================

def get_timestamp():
    return datetime.now().strftime("[%H:%M:%S]")

def check_disk_space(path):
    try:
        p = Path(path)
        if not p.exists():
            p.mkdir(parents=True, exist_ok=True)
        total, used, free = shutil.disk_usage(p)
        free_mb = free // (1024 * 1024)
        if free_mb < MIN_DISK_SPACE_MB:
            return False, free_mb
        return True, free_mb
    except Exception:
        return True, 9999 

def load_api_keys():
    load_dotenv()
    keys_str = os.getenv("GOOGLE_API_KEY")
    if not keys_str: raise ValueError("NO API KEY FOUND! Please check .env file.")
    keys = [k.strip() for k in keys_str.split(',') if k.strip()]
    if not keys: raise ValueError("NO VALID API KEYS FOUND!")
    return keys

def is_image_bw(pil_img, threshold=BW_THRESHOLD):
    img_small = pil_img.resize((100, 100)).convert("HSV")
    score = (np.mean(np.array(img_small)[:, :, 1]) / 255.0) * 100
    return score < threshold

def sanitize_exif(original_path, new_size):
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

# ================= KEY MANAGER =================
class KeyManager:
    def __init__(self, api_keys):
        self.clients = [{"client": genai.Client(api_key=k), "key_mask": k[:4]+"..."} for k in api_keys]
        self.total_keys = len(api_keys)
        self.current_index = 0
        self.lock = asyncio.Lock()

    async def get_client(self):
        async with self.lock:
            if not self.clients:
                raise RuntimeError("ALL_KEYS_EXHAUSTED")
            client_data = self.clients[self.current_index % len(self.clients)]
            self.current_index += 1
            return client_data

    async def ban_current_key(self, client_obj):
        async with self.lock:
            for i, c_data in enumerate(self.clients):
                if c_data["client"] == client_obj:
                    k_mask = c_data["key_mask"]
                    tqdm.write(f"{get_timestamp()} {Colors.RED}💀 Key {k_mask} is depleted (Quota). Removing it!{Colors.RESET}")
                    logging.warning(f"Key {k_mask} removed due to Quota limit.")
                    self.clients.pop(i)
                    return

# ================= API CALL =================

async def try_generate(client, image_bytes, mime_type, prompt, resolution, timeout):
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
    filename = file_path.name
    gc.collect() 

    async with sem:
        try:
            # DISK SPACE CHECK
            has_space, free_mb = check_disk_space(output_dir)
            if not has_space:
                tqdm.write(f"{get_timestamp()} {Colors.RED}💾 DISK FULL! Only {free_mb}MB left. Pausing script!{Colors.RESET}")
                raise RuntimeError("DISK_FULL")

            relative_path = file_path.relative_to(input_base_path)
            target_folder = output_dir / relative_path.parent
            
            with Image.open(file_path) as img:
                is_bw = False
                current_mode = mode
                if mode == "AUTO":
                    is_bw = is_image_bw(img)
                    current_mode = "RESTORE_BW" if is_bw else "RESTORE_COLOR"
                
                jobs = [current_mode]
                if args.auto_colorize_bw and ((mode == "AUTO" and is_bw) or (mode == "RESTORE_BW")):
                    jobs.append("RESTORE_BW_COLORIZE")

                for job_mode in jobs:
                    target_folder.mkdir(parents=True, exist_ok=True)
                    suffix = SUFFIX_MAP.get(job_mode, "-processed")
                    out_filename = f"{args.prefix}{file_path.stem}{suffix}.{args.format.lower()}"
                    out_path = target_folder / out_filename
                    
                    # Double Check (Safety)
                    if args.skip_existing and out_path.exists() and not args.force:
                        if out_path.stat().st_size > 0:
                            stats["skip"] += 1
                            continue
                    
                    success = False
                    attempt_counter = 0
                    
                    while not success:
                        try:
                            client_data = await key_manager.get_client()
                            client = client_data["client"]
                            
                            tqdm.write(f"{get_timestamp()} {Colors.BLUE}☁️  Uploading: {filename} ({job_mode})...{Colors.RESET}")

                            img_byte_arr = io.BytesIO()
                            use_fallback = attempt_counter >= RETRIES_BEFORE_FALLBACK
                            
                            if not use_fallback:
                                img.save(img_byte_arr, format=UPLOAD_FORMAT_PRIMARY)
                                mime_type = f"image/{UPLOAD_FORMAT_PRIMARY.lower()}"
                            else:
                                img.convert("RGB").save(img_byte_arr, format=UPLOAD_FORMAT_FALLBACK, quality=UPLOAD_JPEG_QUALITY)
                                mime_type = f"image/{UPLOAD_FORMAT_FALLBACK.lower()}"
                                tqdm.write(f"{get_timestamp()} {Colors.CYAN}ℹ️  Fallback to JPEG for {filename}{Colors.RESET}")

                            response = await try_generate(
                                client, img_byte_arr.getvalue(), mime_type, 
                                PROMPTS[job_mode], args.resolution, args.timeout
                            )

                            tqdm.write(f"{get_timestamp()} {Colors.GREEN}✨ Processing: {filename}...{Colors.RESET}")

                            if response and response.candidates:
                                candidate = response.candidates[0]
                                if hasattr(candidate, 'content') and candidate.content and candidate.content.parts:
                                    for part in candidate.content.parts:
                                        if part.inline_data:
                                            tqdm.write(f"{get_timestamp()} {Colors.BLUE}💾 Saving: {filename}...{Colors.RESET}")
                                            restored_img = Image.open(io.BytesIO(part.inline_data.data))
                                            exif_bytes = sanitize_exif(str(file_path), restored_img.size)
                                            
                                            save_kwargs = {}
                                            if args.format.upper() in ["JPG", "JPEG"]:
                                                save_kwargs["quality"] = args.quality
                                                save_kwargs["subsampling"] = 0 
                                                if restored_img.mode in ("RGBA", "P"): restored_img = restored_img.convert("RGB")
                                            if exif_bytes: save_kwargs["exif"] = exif_bytes

                                            restored_img.save(out_path, format=args.format, **save_kwargs)
                                            os.utime(out_path, (os.stat(file_path).st_atime, os.stat(file_path).st_mtime))
                                            
                                            restored_img.close()
                                            del restored_img
                                            
                                            stats["success"] += 1
                                            success = True
                                            break 
                                    
                                    if success: break
                                    
                                    msg = f"🛡️ Safety/Content Block by Google for {filename}"
                                    tqdm.write(f"{get_timestamp()} {Colors.YELLOW}{msg}{Colors.RESET}")
                                    logging.warning(msg)
                                    stats["error"] += 1 
                                    stats["failed_files"].append(f"{filename} (Safety Block)")
                                    success = True 
                                else:
                                    msg = f"🛡️ Safety Filter triggered for {filename}"
                                    tqdm.write(f"{get_timestamp()} {Colors.YELLOW}{msg}{Colors.RESET}")
                                    logging.warning(msg)
                                    stats["error"] += 1
                                    stats["failed_files"].append(f"{filename} (Safety Block)")
                                    success = True
                            else:
                                raise Exception("Empty API Response")

                            img_byte_arr.close()
                            del img_byte_arr

                        except Exception as e:
                            error_msg = str(e).lower()
                            if "disk_full" in error_msg: raise e 
                            if "429" in error_msg:
                                await key_manager.ban_current_key(client)
                                continue
                            elif "blocked" in error_msg or "safety" in error_msg or "400" in error_msg:
                                stats["error"] += 1
                                stats["failed_files"].append(f"{filename} (Safety/Blocked)")
                                success = True
                                break
                            elif "all_keys_exhausted" in error_msg:
                                raise RuntimeError("ALL_KEYS_EXHAUSTED")
                            else:
                                attempt_counter += 1
                                wait_time = 2 ** attempt_counter
                                if attempt_counter > MAX_RETRIES_PER_KEY:
                                    stats["error"] += 1
                                    stats["failed_files"].append(f"{filename} (Tech Error)")
                                    return
                                else:
                                    msg = "Server Error 503" if ("503" in error_msg or "deadline" in error_msg) else str(e)
                                    tqdm.write(f"{get_timestamp()} {Colors.YELLOW}⚠️  {filename}: {msg}. Waiting {wait_time}s.{Colors.RESET}")
                                    await asyncio.sleep(wait_time)

        except RuntimeError as e:
            if "ALL_KEYS_EXHAUSTED" in str(e): raise e 
            if "DISK_FULL" in str(e): raise e

        except Exception as e:
            msg = f"System error at {filename}: {e}"
            logging.error(msg)
            tqdm.write(f"{get_timestamp()} {Colors.RED}❌ {msg}{Colors.RESET}")
            stats["error"] += 1
        finally:
            gc.collect()

async def main():
    parser = argparse.ArgumentParser(description="AI Photo Restoration V15 (Efficiency & Overview)")
    parser.add_argument("--input", default=DEFAULT_INPUT_FOLDER, help="Input folder")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_FOLDER, help="Output folder")
    parser.add_argument("--mode", default=DEFAULT_MODE, help="Mode")
    parser.add_argument("--parallel", type=int, default=PARALLEL_LIMIT, help="Threads (Default: 2)")
    parser.add_argument("--format", default=SAVE_FORMAT, help="Save format")
    parser.add_argument("--quality", type=int, default=SAVE_JPEG_QUALITY, help="JPEG Quality")
    parser.add_argument("--skip_existing", type=bool, default=SKIP_EXISTING, help="Skip existing")
    parser.add_argument("--resolution", default=DEFAULT_RESOLUTION, help="Resolution")
    parser.add_argument("--start_delay", type=float, default=DEFAULT_START_DELAY, help="Delay (Default: 0)")
    parser.add_argument("--prefix", default=FILENAME_PREFIX, help="Prefix")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="Timeout")
    parser.add_argument("--auto_colorize_bw", action="store_true", help="Dual-BW")
    parser.add_argument("--force", action="store_true", help="Force")
    
    args = parser.parse_args()
    if not args.auto_colorize_bw: args.auto_colorize_bw = True

    try:
        api_keys = load_api_keys()
        key_manager = KeyManager(api_keys)
        print(f"{Colors.HEADER}--- API SETUP (V15 Efficiency) ---{Colors.RESET}")
        print(f"Found Keys: {len(api_keys)}")
        
        has_space, free_mb = check_disk_space(args.output)
        print(f"Free Disk Space: {Colors.CYAN}{free_mb} MB{Colors.RESET}")
        if not has_space:
             print(f"{Colors.RED}❌ ERROR: Not enough disk space (<{MIN_DISK_SPACE_MB}MB).{Colors.RESET}")
             return

    except ValueError as e:
        print(f"{Colors.RED}Error: {e}{Colors.RESET}")
        return

    in_path = Path(args.input)
    out_path = Path(args.output)
    
    # --- SCANNING & PRE-CHECK ---
    forbidden_suffixes = list(SUFFIX_MAP.values())
    def is_restored_file(path):
        return any(path.stem.lower().endswith(s) for s in forbidden_suffixes)

    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp", ".heic"}
    
    print(f"Scanning '{args.input}' for images...")
    all_source_files = [f for f in in_path.rglob("*") 
                        if f.suffix.lower() in exts and f.is_file() and not is_restored_file(f)]
    
    if not all_source_files:
        print("No images found.")
        return

    # --- FILTERING: Done vs Todo ---
    files_to_process = []
    skipped_count = 0
    
    if args.skip_existing:
        print("Checking for existing restored files...")
        for f in all_source_files:
            relative_path = f.relative_to(in_path)
            target_folder = out_path / relative_path.parent
            base_filename = f"{args.prefix}{f.stem}"
            
            # Smart Check: Look for ANY variation (Color, BW, etc.)
            # If any restored version exists, we consider it "touched/done" for the overview
            found_any = False
            for s in forbidden_suffixes:
                check_path = target_folder / f"{base_filename}{s}.{args.format.lower()}"
                if check_path.exists() and check_path.stat().st_size > 0:
                    found_any = True
                    break
            
            if found_any and not args.force:
                skipped_count += 1
            else:
                files_to_process.append(f)
    else:
        files_to_process = all_source_files

    # --- STATUS TABLE ---
    print(f"\n{Colors.HEADER}=== JOB STATUS ==={Colors.RESET}")
    print(f"📂 Total Images Found: {len(all_source_files)}")
    print(f"✅ Already Restored:   {Colors.GREEN}{skipped_count}{Colors.RESET}")
    print(f"🚀 {Colors.BOLD}Remaining to Process: {len(files_to_process)}{Colors.RESET}")
    print(f"==================\n")

    if not files_to_process:
        print(f"{Colors.GREEN}Nothing left to do! All files are restored.{Colors.RESET}")
        return

    # --- START PROCESSING ---
    stats = { "success": 0, "skip": skipped_count, "error": 0, "failed_files": [] }
    sem = asyncio.Semaphore(args.parallel)
    tasks = []
    
    for f in files_to_process:
        task = asyncio.create_task(process_image_smart(sem, key_manager, f, out_path, in_path, args.mode, args, stats))
        tasks.append(task)
    
    try:
        await tqdm.gather(*tasks, desc="Processing", unit="img")
    except RuntimeError as e:
        if "ALL_KEYS_EXHAUSTED" in str(e):
            now = datetime.now()
            resume_time = now + timedelta(hours=24)
            print(f"\n{Colors.RED}🛑 STOP: Alle API Keys sind aufgebraucht!{Colors.RESET}")
            print(f"Zeitpunkt des Abbruchs: {now.strftime('%d.%m.%Y %H:%M')}")
            print(f"   Voraussichtlicher Neustart möglich ab: {Colors.BOLD}{resume_time.strftime('%d.%m.%Y %H:%M')}{Colors.RESET}")
        elif "DISK_FULL" in str(e):
            print(f"\n{Colors.RED}🛑 STOP: Festplatte ist voll!{Colors.RESET}")
        else:
            print(f"\n{Colors.RED}🛑 Unbekannter Fehler: {e}{Colors.RESET}")

    print(f"\n{Colors.HEADER}STATISTICS{Colors.RESET}")
    print(f"✅ Successful this run: {stats['success']}")
    print(f"⏩ Total Skipped/Done:   {stats['skip']}")
    print(f"❌ Failed:              {stats['error']}")
    if stats["failed_files"]:
        print(f"\n{Colors.YELLOW}⚠️  FAILED FILES:{Colors.RESET}")
        for fail in stats["failed_files"]: print(f"   - {fail}")

if __name__ == "__main__":
    if os.name == 'nt': asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    try: asyncio.run(main())
    except KeyboardInterrupt: print(f"\n{Colors.RED}Aborted by user.{Colors.RESET}")