import argparse
import gc
import json
import os
import shutil
import sys
import numpy as np
import cv2
import pyheif
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms

from sklearn.model_selection import train_test_split

# Dual-branch model (v2: EfficientNet-B0 backbone) and PRNU feature extractor
from model_prnu import DeepFusionNet, EfficientFusionNet, PRNUFusionNet
from prnu_features import extract_prnu_features_fullres
from live_plot import LivePlot

# --- Configuration ---
IMG_WIDTH = 512
IMG_HEIGHT = 512
BATCH_SIZE = 4              # EfficientNet-B0 at 512px fits at batch=4 on 4 GB GPU
GRAD_ACCUM_STEPS = 8        # Effective batch = 4 × 8 = 32
FRAMES_PER_VIDEO = 5        # Number of frames to extract from each video
EPOCHS = 10
PRNU_FEATURE_DIM = 32       # 32-dim fullres PRNU (v3)
AUX_LOSS_WEIGHT  = 0.15     # weight for each auxiliary head loss

# Get the absolute path of the script's directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Construct absolute paths for data and models directories
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
MODELS_DIR = os.path.join(SCRIPT_DIR, '..', 'models')
GENERATED_AI_IMAGES_DIR = os.path.join(SCRIPT_DIR, '..', 'data', 'generated_ai', 'images')
EDITED_AI_IMAGES_DIR = os.path.join(SCRIPT_DIR, '..', 'data', 'edited_ai', 'images')
ARCHIVE_DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data', 'Genimage', 'archive')
ARCHIVE_DATA_DIR_2 = os.path.join(SCRIPT_DIR, '..', 'data', 'Genimage', 'archive(1)')
ARCHIVE_DATA_DIR_3 = os.path.join(SCRIPT_DIR, '..', 'data', 'Genimage', 'archive(2)')
TRAIN_DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data', 'train')

TRAINED_DATA_FILE = os.path.join(MODELS_DIR, 'trained_data.json')
# Model save paths (DeepFusionNet v6)
FUSION_MODEL_PATH    = os.path.join(MODELS_DIR, 'ai_detector_prnu_fusion_v6.pth')
FUSION_MODEL_TS_PATH = os.path.join(MODELS_DIR, 'ai_detector_prnu_fusion_v6_script.ts')
QUANTIZED_MODEL_PATH = os.path.join(MODELS_DIR, 'ai_detector_quantized_int8.pt')

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# ═══════════════════════════════════════════════════════════════════════════
#  Resource Safety Guard
# ═══════════════════════════════════════════════════════════════════════════

class ResourceGuard:
    """
    Monitors system resources and triggers graceful shutdown before the
    system freezes.  Checks: disk space, RAM, GPU VRAM.
    """
    # Thresholds (tune for your laptop)
    MIN_DISK_MB    = 500    # stop if < 500 MB disk free
    MIN_RAM_MB     = 400    # stop if < 400 MB RAM free
    MAX_RAM_PCT    = 92     # stop if RAM > 92% used
    MAX_GPU_PCT    = 95     # stop if GPU VRAM > 95% used

    @staticmethod
    def get_disk_free_mb(path='.'):
        """Get free disk space in MB for the partition containing `path`."""
        try:
            usage = shutil.disk_usage(os.path.abspath(path))
            return usage.free / 1024**2
        except Exception:
            return float('inf')  # assume OK if can't check

    @staticmethod
    def get_ram_free_mb():
        """Get available RAM in MB (Linux /proc/meminfo)."""
        try:
            with open('/proc/meminfo', 'r') as f:
                for line in f:
                    if line.startswith('MemAvailable:'):
                        return int(line.split()[1]) / 1024
            return float('inf')
        except Exception:
            return float('inf')

    @staticmethod
    def get_ram_percent():
        """Get RAM usage percentage."""
        try:
            with open('/proc/meminfo', 'r') as f:
                info = {}
                for line in f:
                    parts = line.split()
                    info[parts[0].rstrip(':')] = int(parts[1])
            total = info.get('MemTotal', 1)
            avail = info.get('MemAvailable', total)
            return ((total - avail) / total) * 100
        except Exception:
            return 0

    @staticmethod
    def get_gpu_percent():
        """Get GPU VRAM usage percentage."""
        try:
            if torch.cuda.is_available():
                alloc = torch.cuda.memory_allocated()
                total = torch.cuda.get_device_properties(0).total_memory
                return (alloc / total) * 100 if total > 0 else 0
        except Exception:
            pass
        return 0

    @classmethod
    def check(cls, context=''):
        """
        Check all resources. Returns (is_safe, reason).
        Call this periodically during training.
        """
        # Disk
        disk_mb = cls.get_disk_free_mb(MODELS_DIR)
        if disk_mb < cls.MIN_DISK_MB:
            return False, f"Disk space critically low: {disk_mb:.0f} MB free (min {cls.MIN_DISK_MB} MB)"

        # RAM
        ram_mb = cls.get_ram_free_mb()
        if ram_mb < cls.MIN_RAM_MB:
            return False, f"RAM critically low: {ram_mb:.0f} MB free (min {cls.MIN_RAM_MB} MB)"

        ram_pct = cls.get_ram_percent()
        if ram_pct > cls.MAX_RAM_PCT:
            return False, f"RAM usage too high: {ram_pct:.0f}% (max {cls.MAX_RAM_PCT}%)"

        # GPU
        gpu_pct = cls.get_gpu_percent()
        if gpu_pct > cls.MAX_GPU_PCT:
            return False, f"GPU VRAM too high: {gpu_pct:.0f}% (max {cls.MAX_GPU_PCT}%)"

        return True, ''

    @classmethod
    def check_or_abort(cls, model, context=''):
        """
        Check resources. If critical, save model and exit gracefully.
        Returns True if safe to continue.
        """
        safe, reason = cls.check(context)
        if not safe:
            print(f"\n  🛑  EMERGENCY STOP: {reason}")
            print(f"       Context: {context}")
            print(f"       Saving model before exit...")
            try:
                os.makedirs(MODELS_DIR, exist_ok=True)
                emergency_path = os.path.join(MODELS_DIR, 'ai_detector_model_pytorch_EMERGENCY.pth')
                torch.save(model.state_dict(), emergency_path)
                print(f"  💾  Emergency model saved: {emergency_path}")
            except Exception as e:
                print(f"  ⚠️  Could not save emergency model: {e}")

            # Free everything we can
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print(f"\n  ❌  Training stopped safely to prevent system freeze.")
            print(f"      Free up resources and re-run with --resume to continue.\n")
            sys.exit(1)
        return True

def load_heic(path):
    """
    Loads a HEIC image using pyheif and converts it to a PIL Image.
    """
    try:
        heif_file = pyheif.read(path)
        # Assuming there's at least one image in the HEIC file
        pi = Image.frombytes(
            heif_file.mode, 
            heif_file.size, 
            heif_file.data,
            "raw",
            heif_file.mode,
            heif_file.stride,
        )
        return pi.convert("RGB")
    except Exception as e:
        print(f"Error processing HEIC file {path} with pyheif: {e}")
        return None

def extract_frames_from_video(video_path, num_frames=FRAMES_PER_VIDEO):
    """
    Extracts a fixed number of frames from a video file using OpenCV and returns them as PIL Images.
    """
    frames = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return frames

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0

    if total_frames == 0 or duration == 0:
        print(f"Warning: Video {video_path} has zero frames or duration.")
        cap.release()
        return frames

    # Calculate frame indices to extract
    # Ensure num_frames doesn't exceed total_frames
    actual_frames_to_extract = min(num_frames, total_frames)
    
    # Extract frames evenly spaced
    if actual_frames_to_extract > 0:
        indices = np.linspace(0, total_frames - 1, actual_frames_to_extract, dtype=int)
    else:
        indices = []

    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            print(f"Warning: Could not read frame {i} from {video_path}")
            continue
        # OpenCV reads frames as BGR, convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame_rgb))
    
    cap.release()
    if not frames:
        print(f"Warning: No frames were successfully extracted from {video_path}.")
    return frames

class AIDetectorDataset(Dataset):
    """
    Dataset that returns (image_tensor, prnu_features, label).

    `prnu_features` is an 8-dim float32 vector extracted from the image
    by prnu_features.extract_prnu_features().  The PRNUFusionNet model
    consumes both the image tensor and the PRNU features during training.
    """
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # --- Load image ---
        if img_path.lower().endswith('.heic'):
            image = load_heic(img_path)
            if image is None:
                print(f"Skipping problematic HEIC file: {img_path}")
                image = Image.new('RGB', (IMG_WIDTH, IMG_HEIGHT), color='black')
        else:
            try:
                image = Image.open(img_path).convert('RGB')
            except Exception as e:
                print(f"  Warning: could not load {img_path}: {e} — using blank frame")
                image = Image.new('RGB', (IMG_WIDTH, IMG_HEIGHT), color='black')

        # --- Extract PRNU features (16-dim fullres) ---
        try:
            prnu_feats = extract_prnu_features_fullres(image)   # (16,)
            if not np.isfinite(prnu_feats).all():
                prnu_feats = np.zeros(PRNU_FEATURE_DIM, dtype=np.float32)
        except Exception as e:
            print(f"  PRNU feature extraction failed for {img_path}: {e}")
            prnu_feats = np.zeros(PRNU_FEATURE_DIM, dtype=np.float32)

        # --- Apply image transform (resize, normalize, augment) ---
        if self.transform:
            image = self.transform(image)

        prnu_tensor = torch.from_numpy(prnu_feats).float()
        return image, prnu_tensor, torch.tensor(label, dtype=torch.float32)

def collect_data_paths(previously_trained_paths, image_data_dir=None):
    """
    Collects image and video paths from various directories, filtering out previously trained original files.
    Returns:
        A tuple of (list of file paths for current training, list of labels, list of newly trained original file paths).
    """
    all_file_paths = [] # Paths including temporary video frames for the current training session
    all_labels = []
    newly_trained_original_paths = [] # Only original image/video files, not temp frames

    if image_data_dir:
        print(f"Collecting images from specified directory: {image_data_dir}")
        if not os.path.exists(image_data_dir):
            print(f"Error: Specified image data directory not found: {image_data_dir}")
            return [], [], []

        file_list = []
        for root, _, files in os.walk(image_data_dir):
            file_list.extend([os.path.join(root, f) for f in files])
        
        total_files = len(file_list)
        print(f"  → Scanning {total_files} files...")

        for i, file_path in enumerate(file_list):
            if (i + 1) % 10000 == 0:
                print(f"    ... processed {i + 1}/{total_files} files")
            
            if file_path in previously_trained_paths:
                continue

            ext = os.path.splitext(file_path)[1].lower()
            if ext in ['.jpg', '.jpeg', '.png', '.heic']:
                newly_trained_original_paths.append(file_path)
                # For a specified directory, we don't know the label (AI or Real).
                # For simplicity, we'll assume all images in the specified directory are "Real" (label 0)
                # or "AI" (label 1) based on a naming convention or a prompt to the user to classify.
                # For now, let's assume all images in a user-provided directory are AI (label 1) by default,
                # as the primary use case is likely to train against new AI-generated content.
                # If the user wants to specify both AI and Real from a custom path, they would need two separate calls or a more complex directory structure.
                # For now, let's keep it simple and assume they are all AI if from a custom path, or we can prompt the user.
                
                # For now, I'll prompt the user if they want to classify the images as AI or Real.
                all_file_paths.append(file_path)
                # We need to decide how to label these. For a generic `image_data_dir`, we can't assume.
                # I will make a decision to consider images from user-specified path as `AI` generated, since the primary purpose of this tool is to detect AI images.
                # If a user wants to train with 'real' images from a custom path, they should use the default behavior by placing them in `data/real`.
                all_labels.append(1) # Default to AI for custom provided images
            else:
                pass # Already skipped or warned
        return all_file_paths, all_labels, newly_trained_original_paths

    # --- Original data collection logic (if image_data_dir is not provided) ---

    else:
        # --- Original data collection logic (if image_data_dir is not provided) ---
        real_data_categories = ['real', 'personal']
        for category in real_data_categories:
            category_path = os.path.join(DATA_DIR, category)
            if not os.path.exists(category_path):
                print(f"Warning: Directory not found: {category_path}")
                continue

            print(f"Collecting {category} content from {category_path}")
            filenames = os.listdir(category_path)
            total_files = len(filenames)
            for i, filename in enumerate(filenames):
                if (i + 1) % 5000 == 0:
                    print(f"    ... processed {i + 1}/{total_files} in {category}")
                
                file_path = os.path.join(category_path, filename)

                if file_path in previously_trained_paths:
                    continue

                ext = os.path.splitext(filename)[1].lower()

                if ext in ['.jpg', '.jpeg', '.png', '.heic']:
                    newly_trained_original_paths.append(file_path)
                    all_file_paths.append(file_path)
                    all_labels.append(0)  # Label as Real
                elif ext in ['.mp4', '.mov']:
                    print(f"Processing video {filename} for frame extraction (Label: 0 - Real)")
                    newly_trained_original_paths.append(file_path)
                    frames = extract_frames_from_video(file_path)
                    for j, _ in enumerate(frames):
                        frame_filename = f"{os.path.splitext(filename)[0]}_frame{j:03d}.png"
                        temp_frame_path = os.path.join(SCRIPT_DIR, '..', 'data', 'temp_frames', frame_filename)
                        os.makedirs(os.path.dirname(temp_frame_path), exist_ok=True)
                        frames[j].save(temp_frame_path)
                        all_file_paths.append(temp_frame_path)
                        all_labels.append(0)  # Label as Real
                else:
                    pass

        # --- Load from data/ai (label 1) ---
        ai_data_categories = ['ai']
        for category in ai_data_categories:
            category_path = os.path.join(DATA_DIR, category)
            if not os.path.exists(category_path):
                print(f"Warning: Directory not found: {category_path}")
                continue

            print(f"Collecting {category} content from {category_path}")
            filenames = os.listdir(category_path)
            total_files = len(filenames)
            for i, filename in enumerate(filenames):
                if (i + 1) % 5000 == 0:
                    print(f"    ... processed {i + 1}/{total_files} in {category}")

                file_path = os.path.join(category_path, filename)

                if file_path in previously_trained_paths:
                    continue

                ext = os.path.splitext(filename)[1].lower()

                if ext in ['.jpg', '.jpeg', '.png', '.heic']:
                    newly_trained_original_paths.append(file_path)
                    all_file_paths.append(file_path)
                    all_labels.append(1)  # Label as AI
                else:
                    pass

        # --- Load from generated_ai/images (always label 1 for AI) ---
        if os.path.exists(GENERATED_AI_IMAGES_DIR):
            print(f"Collecting AI-generated images from {GENERATED_AI_IMAGES_DIR}")
            for filename in os.listdir(GENERATED_AI_IMAGES_DIR):
                file_path = os.path.join(GENERATED_AI_IMAGES_DIR, filename)

                if file_path in previously_trained_paths:
                    print(f"Skipping previously trained file: {file_path}")
                    continue

                ext = os.path.splitext(filename)[1].lower()
                if ext in ['.jpg', '.jpeg', '.png', '.heic']:
                    newly_trained_original_paths.append(file_path)
                    all_file_paths.append(file_path)
                    all_labels.append(1)  # Label as AI
                else:
                    print(f"Skipping unsupported file type: {filename} in {GENERATED_AI_IMAGES_DIR}")
        else:
            print(f"Warning: Directory not found: {GENERATED_AI_IMAGES_DIR}")

        # --- Load from edited_ai/images (always label 1 for AI) ---
        if os.path.exists(EDITED_AI_IMAGES_DIR):
            print(f"Collecting AI-edited images from {EDITED_AI_IMAGES_DIR}")
            for filename in os.listdir(EDITED_AI_IMAGES_DIR):
                file_path = os.path.join(EDITED_AI_IMAGES_DIR, filename)

                if file_path in previously_trained_paths:
                    print(f"Skipping previously trained file: {file_path}")
                    continue

                ext = os.path.splitext(filename)[1].lower()
                if ext in ['.jpg', '.jpeg', '.png', '.heic']:
                    newly_trained_original_paths.append(file_path)
                    all_file_paths.append(file_path)
                    all_labels.append(1)  # Label as AI
                else:
                    print(f"Skipping unsupported file type: {filename} in {EDITED_AI_IMAGES_DIR}")
        else:
            print(f"Warning: Directory not found: {EDITED_AI_IMAGES_DIR}")

        # --- Load from data/train/REAL (label 0) and data/train/FAKE (label 1) ---
        train_label_map = {'REAL': 0, 'FAKE': 1}
        for subfolder, label in train_label_map.items():
            subfolder_path = os.path.join(TRAIN_DATA_DIR, subfolder)
            if not os.path.exists(subfolder_path):
                print(f"Warning: Train subdirectory not found: {subfolder_path}")
                continue

            label_name = 'Real' if label == 0 else 'AI/Fake'
            print(f"Collecting {label_name} content from {subfolder_path}")
            count = 0
            file_list = []
            for root, _, files in os.walk(subfolder_path):
                file_list.extend([os.path.join(root, f) for f in files])
            
            total_files = len(file_list)
            print(f"  → Scanning {total_files} files...")
            
            for i, file_path in enumerate(file_list):
                if (i + 1) % 10000 == 0:
                    print(f"    ... processed {i + 1}/{total_files} files")
    
                if file_path in previously_trained_paths:
                    continue
    
                ext = os.path.splitext(file_path)[1].lower()
                if ext in ['.jpg', '.jpeg', '.png', '.heic']:
                    newly_trained_original_paths.append(file_path)
                    all_file_paths.append(file_path)
                    all_labels.append(label)
                    count += 1
            print(f"  → Collected {count} new {label_name} images from {subfolder_path}")

        if not all_file_paths:
            print("Error: No data paths were collected. Please check the data directories and file formats.")
            return [], [], []

        return all_file_paths, all_labels, newly_trained_original_paths

# PRNUFusionNet is imported from model_prnu at the top of the file.
# The old PyTorchCNN class is retained inside model_prnu.py as a shim.

def train_pytorch_model(image_data_dir=None, use_prnu=True,
                        resume_path=None, num_workers=2, epochs=EPOCHS,
                        plot=True):
    """
    Trains the PyTorch model.
    """
    print("\n--- Collecting Data Paths ---")
    
    # Load previously trained paths
    previously_trained_paths = set()
    if os.path.exists(TRAINED_DATA_FILE):
        with open(TRAINED_DATA_FILE, 'r') as f:
            try:
                previously_trained_paths = set(json.load(f))
                print(f"Loaded {len(previously_trained_paths)} previously trained paths.")
            except json.JSONDecodeError:
                print("Warning: trained_data.json is empty or malformed. Starting with fresh tracking.")
    
    image_paths, labels, newly_trained_original_paths = collect_data_paths(previously_trained_paths, image_data_dir)

    if not image_paths:
        print("No NEW data available for training. Exiting.")
        return

    # Split data for training and validation (80/20 split)
    X_train_paths, X_val_paths, y_train, y_val = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42
    )

    # Define transformations — augmentation for training only
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop((IMG_HEIGHT, IMG_WIDTH), scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Validation transform — deterministic, no augmentation
    val_transform = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create datasets and dataloaders
    train_dataset = AIDetectorDataset(X_train_paths, y_train, transform=train_transform)
    val_dataset = AIDetectorDataset(X_val_paths, y_val, transform=val_transform)

    # Balanced sampling: give each sample a weight inverse to its class frequency
    y_train_arr = np.array(y_train)
    num_real  = int((y_train_arr == 0).sum())
    num_ai    = int((y_train_arr == 1).sum())
    print(f"  Train class balance — REAL: {num_real}, AI: {num_ai}")
    class_weights = {0: 1.0 / max(num_real, 1), 1: 1.0 / max(num_ai, 1)}
    sample_weights = [class_weights[int(l)] for l in y_train_arr]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=num_workers, pin_memory=(num_workers > 0))
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=num_workers, pin_memory=(num_workers > 0))

    print(f"Training with {len(train_dataset)} samples, validating with {len(val_dataset)} samples.")
    print(f"PRNU features enabled: {use_prnu}")

    # Initialize model — DeepFusionNet v6 (EfficientNet-B3 + 11 forensic branches)
    model = DeepFusionNet(prnu_in_features=PRNU_FEATURE_DIM).to(device)
    model.param_summary()

    # pos_weight balances the loss: downweight the majority AI class
    y_all = np.array(labels)
    pw = float((y_all == 0).sum()) / max(float((y_all == 1).sum()), 1)
    pos_weight = torch.tensor([pw], dtype=torch.float32).to(device)
    print(f"  BCEWithLogitsLoss pos_weight = {pw:.4f}")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Only pass trainable (unfrozen) parameters to the optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"  Passing {sum(p.numel() for p in trainable_params):,} trainable params to optimizer")

    # AdamW + weight decay for better generalisation (vs plain Adam)
    optimizer = optim.AdamW(trainable_params, lr=1e-3, weight_decay=1e-4)

    # OneCycleLR: ramps up then anneals — fastest convergence
    steps_per_epoch = max(len(train_loader) // GRAD_ACCUM_STEPS, 1)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=1e-3,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        pct_start=0.3,
    )

    # Mixed precision for GPU — halves VRAM usage
    use_amp = (device.type == 'cuda')
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    # ── Resume from checkpoint if requested ──────────────────────────────
    start_epoch = 0
    if resume_path and os.path.exists(resume_path):
        print(f"\nResuming from checkpoint: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt.get('epoch', 0)
        print(f"  Resuming from epoch {start_epoch + 1}/{epochs}")
    elif resume_path:
        print(f"  Warning: checkpoint not found at {resume_path}, starting fresh.")

    live_plot = LivePlot(title='Image Training — DeepFusionNet (local data)', xlabel='Epoch') if plot else None

    print("\n--- Starting PyTorch Model Training ---")
    for epoch in range(start_epoch, epochs):
        # ── Safety check at epoch start ──
        ResourceGuard.check_or_abort(model, f"Epoch {epoch+1}/{epochs} start")

        model.train() # Set model to training mode
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        optimizer.zero_grad()

        for i, (inputs, prnu_feats, targets) in enumerate(train_loader):
            inputs     = inputs.to(device)
            prnu_feats = prnu_feats.to(device)
            targets    = targets.to(device).view(-1, 1)

            # Forward pass with AMP — dual-branch input
            with torch.amp.autocast('cuda', enabled=use_amp):
                outputs = model(inputs, prnu_feats)   # DeepFusionNet v6

                # Unpack: train mode returns (logit, *aux_logits)
                if isinstance(outputs, tuple):
                    logit      = outputs[0]            # (B, 1)
                    aux_logits = outputs[1:]           # 7 aux logit tensors
                else:
                    logit      = outputs
                    aux_logits = []

                # Primary loss
                primary_loss = criterion(logit, targets)

                # Auxiliary losses (each aux head also predicts binary AI/real)
                aux_loss = sum(
                    criterion(a, targets) for a in aux_logits
                ) * AUX_LOSS_WEIGHT if aux_logits else torch.tensor(0.0, device=device)

                loss = (primary_loss + aux_loss) / GRAD_ACCUM_STEPS

            # Check for NaN loss
            if torch.isnan(loss):
                print(f"  ⚠️  WARNING: NaN loss detected at batch {i+1}. Skipping batch.")
                del inputs, prnu_feats, targets, outputs, loss, logit, aux_logits
                optimizer.zero_grad()
                continue

            # Backward with scaler
            scaler.scale(loss).backward()

            # Step every GRAD_ACCUM_STEPS micro-batches
            if (i + 1) % GRAD_ACCUM_STEPS == 0:
                # Gradient clipping — prevents exploding gradients
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            running_loss += loss.item() * GRAD_ACCUM_STEPS * inputs.size(0)

            predicted = (torch.sigmoid(logit.detach()) > 0.5).float()
            total_samples += targets.size(0)
            correct_predictions += (predicted == targets).sum().item()

            if (i + 1) % 10 == 0:
                print(f"    Batch {i+1}/{len(train_loader)} | Loss: {running_loss / (total_samples if total_samples > 0 else 1):.4f} | Acc: {correct_predictions / (total_samples if total_samples > 0 else 1):.4f}")

            # Free GPU tensors immediately
            del inputs, prnu_feats, targets, outputs, loss, logit, aux_logits

            # ── Safety check every 50 batches ──
            if (i + 1) % 50 == 0:
                ResourceGuard.check_or_abort(model, f"Epoch {epoch+1} batch {i+1}")

        # Flush remaining gradients if not already flushed by the last batch
        if (len(train_loader) % GRAD_ACCUM_STEPS) != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        epoch_loss = running_loss / max(len(train_dataset), 1)
        epoch_acc  = correct_predictions / max(total_samples, 1)
        current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else optimizer.param_groups[0]['lr']

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}, LR: {current_lr:.6f}")

        # Validation phase
        model.eval() # Set model to evaluation mode
        val_running_loss = 0.0
        val_correct_predictions = 0
        val_total_samples = 0
        with torch.no_grad():
            for inputs, prnu_feats, targets in val_loader:
                inputs     = inputs.to(device)
                prnu_feats = prnu_feats.to(device)
                targets    = targets.to(device).view(-1, 1)

                outputs = model(inputs, prnu_feats)   # dual-branch
                # Unpack eval output (always just logit in eval mode)
                logit = outputs[0] if isinstance(outputs, tuple) else outputs
                loss = criterion(logit, targets)

                val_running_loss += loss.item() * inputs.size(0)

                predicted = (torch.sigmoid(logit) > 0.5).float()
                val_total_samples += targets.size(0)
                val_correct_predictions += (predicted == targets).sum().item()
        
        val_loss = val_running_loss / max(len(val_dataset), 1)
        val_acc = val_correct_predictions / max(val_total_samples, 1)
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

        if live_plot:
            live_plot.update(epoch + 1, epoch_loss, epoch_acc, val_loss, val_acc)

        # Save latest checkpoint after every epoch (essential for Colab)
        os.makedirs(MODELS_DIR, exist_ok=True)
        ckpt_path = os.path.join(MODELS_DIR, 'checkpoint_latest.pth')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'val_acc': val_acc,
        }, ckpt_path)
        print(f"Checkpoint saved → {ckpt_path}")

        # Free memory between epochs
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    print("\n--- PyTorch Model Training Finished ---")

    # Save the trained DeepFusionNet v6
    os.makedirs(MODELS_DIR, exist_ok=True)
    torch.save(model.state_dict(), FUSION_MODEL_PATH)
    print(f"DeepFusionNet v6 (state_dict) saved to: {FUSION_MODEL_PATH}")

    # Save TorchScript for mobile (C++ / Android / iOS)
    model.eval()
    with torch.no_grad():
        dummy_img  = torch.randn(1, 3, IMG_HEIGHT, IMG_WIDTH).to(device)
        dummy_prnu = torch.zeros(1, PRNU_FEATURE_DIM).to(device)
        traced_model = torch.jit.trace(model, (dummy_img, dummy_prnu))
        traced_model.save(FUSION_MODEL_TS_PATH)
        del dummy_img, dummy_prnu, traced_model
    print(f"EfficientFusionNet (TorchScript) saved to: {FUSION_MODEL_TS_PATH}")

    # --- Export quantized int8 model for CPU inference (8x smaller) ---
    print("\n--- Exporting int8 quantized model ---")
    try:
        model_cpu = EfficientFusionNet()
        model_cpu.load_state_dict(torch.load(FUSION_MODEL_PATH, map_location='cpu'))
        model_cpu.eval().cpu()
        quantized = torch.quantization.quantize_dynamic(
            model_cpu,
            {nn.Linear},          # quantize only Linear layers (safe for conv+linear models)
            dtype=torch.qint8
        )
        torch.save(quantized, QUANTIZED_MODEL_PATH)  # save full quantized object (not just state_dict)
        q_size = os.path.getsize(QUANTIZED_MODEL_PATH) / 1024**2
        f_size = os.path.getsize(FUSION_MODEL_PATH) / 1024**2
        print(f"  ✅ Original : {f_size:.1f} MB")
        print(f"  ✅ Quantized: {q_size:.1f} MB  ({f_size/q_size:.1f}x smaller)")
        print(f"  Saved to: {QUANTIZED_MODEL_PATH}")
        del model_cpu, quantized
    except Exception as e:
        print(f"  ⚠️  Quantization export failed (non-fatal): {e}")

    # Save updated list of trained data paths
    all_trained_paths = previously_trained_paths.union(set(newly_trained_original_paths))
    with open(TRAINED_DATA_FILE, 'w') as f:
        json.dump(list(all_trained_paths), f, indent=4)
    print(f"Updated trained data paths saved to: {TRAINED_DATA_FILE} (Total: {len(all_trained_paths)})")

    # Clean up temporary frames
    temp_frames_dir = os.path.join(SCRIPT_DIR, '..', 'data', 'temp_frames')
    if os.path.exists(temp_frames_dir):
        import shutil
        shutil.rmtree(temp_frames_dir)
        print(f"Cleaned up temporary frames directory: {temp_frames_dir}")


def main():
    """
    The main function to run the PyTorch training pipeline.
    """
    parser = argparse.ArgumentParser(description="AI Detector DeepFusionNet Training Pipeline")
    parser.add_argument('--image_data_dir', type=str, default=None,
                        help='Optional: custom directory for image training data.')
    parser.add_argument('--no-prnu', action='store_true',
                        help='Disable PRNU feature branch (image-only CNN for comparison).')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from a checkpoint .pth file (e.g. models/checkpoint_latest.pth).')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='DataLoader worker processes (use 0 on Colab if multiprocessing errors occur).')
    parser.add_argument('--epochs', type=int, default=EPOCHS,
                        help=f'Number of training epochs (default: {EPOCHS}).')
    parser.add_argument('--plot', action='store_true', default=True,
                        help='Show live loss/accuracy plot (default: on)')
    parser.add_argument('--no_plot', dest='plot', action='store_false',
                        help='Disable live plot (headless/SSH environments)')
    args = parser.parse_args()

    print("--- AI Detector DeepFusionNet Training Pipeline ---")
    train_pytorch_model(
        args.image_data_dir,
        use_prnu=not args.no_prnu,
        resume_path=args.resume,
        num_workers=args.num_workers,
        epochs=args.epochs,
        plot=args.plot,
    )
    print("\n--- Pipeline execution finished. ---")

if __name__ == '__main__':
    main()
