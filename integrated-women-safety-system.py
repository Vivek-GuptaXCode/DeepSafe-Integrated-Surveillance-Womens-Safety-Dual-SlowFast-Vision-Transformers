#!/usr/bin/env python
# coding: utf-8

# # Integrated Women Safety Surveillance System
# 
# ## Violence Against Women Detection using Dual Gender Classification & Dual SlowFast Violence Detection
# 
# This notebook implements the complete 5-stage pipeline:
# 
# 1. **Person Detection & Tracking** — YOLOv5s + IoU-based tracking
# 2. **Visibility-Conditioned Dual-Model Gender Classification** — Two EVA-02 Large ViTs (body + face)
# 3. **Dual-Stream SlowFast Violence Detection** — RGB + Optical Flow SlowFast-R50 networks
# 4. **Grad-CAM Spatial Evidence Localization** — Heatmaps on RGB SlowFast slow pathway
# 5. **Rule-Based Alert Generation** — Lone-woman & violence-against-women alerts with evidence capture
# 
# $$I_t \xrightarrow{\text{YOLO}} \mathcal{D}_t \xrightarrow{\text{IoU Track}} \mathcal{P}_t \xrightarrow[\text{Violence}]{\text{Gender}} (g_i, v_t) \xrightarrow{\text{Grad-CAM}} \mathcal{B}_t \xrightarrow{\text{Alert}} (\ell_t, \mathcal{E}_t)$$

# ## Part 1: Install Dependencies

# In[1]:


get_ipython().system('pip install timm pytorchvideo ultralytics mediapipe opencv-python-headless -q')


# ## Part 2: Imports & Device Setup

# In[2]:


import os
import sys
import cv2
import json
import time
import random
import threading
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import deque
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import timm
import mediapipe as mp
from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for video processing

# Set seed for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

if torch.cuda.is_available():
    # GPU Optimizations
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')
    try:
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
    except Exception:
        pass
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
    print(f'TF32 enabled: {torch.backends.cuda.matmul.allow_tf32}')
    print(f'BFloat16 supported: {torch.cuda.is_bf16_supported()}')
else:
    print('WARNING: CUDA not available. Running on CPU will be very slow.')


# ## Part 3: Configuration

# In[3]:


# ════════════════════════════════════════════════════════════════════
# CONFIGURATION — Adjust paths and parameters here
# ════════════════════════════════════════════════════════════════════

CONFIG = {
    # ── Model Paths ──
    'gender_body_model_path': '/kaggle/input/models/vivekguptaxcode/face-based-gender-classifier-eva-02-large/pytorch/default/1/model_outputs/best_model_phase2.pth',       # Body-based gender classifier
    'gender_face_model_path': '/kaggle/input/models/vivekguptaxcode/gender-classification-face-based-eva-02-large/pytorch/default/1/best_face_model_acc.pth',  # Face-based gender classifier
    'violence_model_path': '/kaggle/input/models/baldbuffe/dual-slowfast-network/pytorch/default/1/best_model.pth',                 # Dual SlowFast violence detector

    # ── Input/Output (Kaggle paths) ──
    'input_video': '/kaggle/input/datasets/vivekguptaxcode/test-video/Bengaluru_News_Bengaluru_Woman_Assaulted_In_Broad_Daylight_CCTV_Captures_Attack_720P.mp4',                              # Path to input video
    'output_video': '/kaggle/working/output_annotated.mp4',        # Annotated output video
    'evidence_dir': '/kaggle/working/evidence_output',             # Evidence capture directory
    'output_dir': '/kaggle/working',                               # General output directory

    # ── Person Detection (YOLOv5s) ──
    'yolo_model': 'yolov5su.pt',    # Ultralytics YOLOv5s model
    'det_confidence': 0.40,         # Detection confidence threshold
    'max_detections': 10,           # Maximum detections per frame

    # ── IoU Tracking ──
    'iou_threshold': 0.50,          # IoU matching threshold
    'track_age_out': 5,             # Frames before track termination

    # ── Visibility (MediaPipe Pose) ──
    'pose_complexity': 0,           # MediaPipe Pose complexity (0=fastest)
    'pose_min_confidence': 0.3,     # Minimum detection confidence
    'lower_body_visibility_threshold': 0.5,  # Keypoint visibility threshold
    'min_visible_lower_landmarks': 4,        # Min landmarks for full-body

    # ── Gender Classification (EVA-02 Large) ──
    'gender_image_size': 448,       # Input size for gender models
    'face_crop_ratio': 0.40,        # Top 40% for face model
    'gender_mean': [0.4850, 0.4560, 0.4060],  # PA-100K normalization
    'gender_std': [0.2290, 0.2240, 0.2250],

    # ── Gender Fusion ──
    'body_weight_full': 0.6,        # Body model weight when full body visible
    'face_weight_full': 0.4,        # Face model weight when full body visible
    'gender_history_len': 5,        # Temporal smoothing window

    # ── Violence Detection (Dual SlowFast) ──
    'violence_input_size': 224,     # Input size for violence models
    'clip_len': 16,                 # Must match training clip length exactly
    'frame_buffer_size': 16,        # Rolling buffer length (kept equal to clip_len)
    'slow_frames': 8,               # Slow pathway temporal samples
    'fast_frames': 32,              # Fast pathway temporal samples (training repeats indices from 16 frames)
    'flow_clip_value': 20.0,        # Match training-time optical flow clipping before uint8 encoding
    'rgb_mean': [0.45, 0.45, 0.45],
    'rgb_std': [0.225, 0.225, 0.225],
    'flow_mean': [0.5, 0.5],
    'flow_std': [0.5, 0.5],
    'violence_run_interval': 2,     # More frequent checks for short violent bursts
    'fight_index': 0,               # Auto-calibrated from dataset if enabled
    'enable_label_calibration': True,
    'calibration_root': '/kaggle/input/rwf2000-opt-rgb/npy/train',
    'calibration_root_candidates': [
        '/kaggle/input/datasets/slimshady7/rwf2000-opt-rgb/npy/train/',
        '/kaggle/input/RWF2000-OPT-RGB/npy/train',
        '/kaggle/input/rwf2000-opt-rgb/npy/Train',
        '/kaggle/input/RWF2000-OPT-RGB/npy/Train'
    ],
    'calibration_samples_per_class': 24,

    # ── Violence Temporal Smoothing ──
    'violence_confidence_threshold': 0.30,
    'violence_temporal_window': 5,
    'violence_min_count': 1,        # Min Fight predictions in window

    # ── Grad-CAM ──
    'gradcam_target_layer': 'blocks.5',  # Slow pathway blocks[5]
    'gradcam_threshold': 0.50,
    'gradcam_min_area': 1600,       # Min contour area (40x40)
    'gradcam_alpha': 0.45,          # Overlay blend alpha

    # ── Alert System ──
    'night_start': 22,              # Night hours start (24h)
    'night_end': 6,                 # Night hours end (24h)
    'violence_persistence_threshold': 2,  # Violence count for alert
    'lone_woman_cooldown': 30,      # Seconds
    'violence_alert_cooldown': 10,  # Seconds

    # ── Evidence Capture ──
    'evidence_confidence': 0.55,
    'evidence_cooldown': 15,        # Seconds
    'evidence_clip_duration': 5,    # Seconds
}

# Create output directories on Kaggle
os.makedirs(CONFIG['output_dir'], exist_ok=True)
os.makedirs(CONFIG['evidence_dir'], exist_ok=True)

print('Configuration loaded. All outputs will be saved to /kaggle/working/')
for k, v in CONFIG.items():
    print(f'  {k}: {v}')


# ## Part 4: Model Architecture Definitions

# ### 4.1 Gender Classifier Architecture (EVA-02 Large ViT)

# In[4]:


# ═══════════════════════════════════════════════════════════════════════
# Gender Classifier: EVA-02 Large ViT-L/14 with Custom Head
# Two separate models: body (full crop) and face (upper 40% crop)
# Architecture identical, trained on different image regions
# ═══════════════════════════════════════════════════════════════════════

class CustomHeadViT(nn.Module):
    """Custom classification head for EVA-02 ViT.
    LayerNorm(1024) → Dropout(0.5) → Linear(1024, 2)
    """
    def __init__(self, num_ftrs, drop_rate=0.5):
        super().__init__()
        self.norm = nn.LayerNorm(num_ftrs)
        self.drop = nn.Dropout(drop_rate)
        self.fc = nn.Linear(num_ftrs, 2)

    def forward(self, x):
        x = self.norm(x)
        x = self.drop(x)
        return self.fc(x)


def build_gender_model(weights_path, device):
    """Build EVA-02 Large gender classifier and load trained weights.
    
    Architecture:
        - Backbone: eva02_large_patch14_448 (304M params)
        - Head: LayerNorm → Dropout(0.5) → Linear(1024, 2)
        - Classes: Female=0, Male=1
    """
    model = timm.create_model(
        'eva02_large_patch14_448.mim_m38m_ft_in22k_in1k',
        pretrained=False,  # We load our own weights
        num_classes=2,
        drop_path_rate=0.4
    )
    
    # Replace head with custom head (must match training architecture)
    num_ftrs = model.head.in_features  # 1024
    model.head = CustomHeadViT(num_ftrs, drop_rate=0.5)
    
    # Load trained weights (handle torch.compile() prefix if present)
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
        print('  Detected torch.compile() checkpoint — stripping _orig_mod. prefix...')
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    print(f'  Loaded gender model from {weights_path}')
    print(f'  Parameters: {sum(p.numel() for p in model.parameters()):,}')
    return model


print('Gender classifier architecture defined (CustomHeadViT + EVA-02 Large).')


# ### 4.2 Violence Detector Architecture (Dual SlowFast-R50)

# In[5]:


# ═══════════════════════════════════════════════════════════════════════
# Violence Detector: Dual SlowFast-R50 (RGB + Optical Flow streams)
# Late fusion via concatenation + linear classifier
# Classes: Fight=0, NonFight=1
# ═══════════════════════════════════════════════════════════════════════

def strip_compiled_prefix(state_dict):
    """Strip '_orig_mod.' prefix from state dict keys.
    
    torch.compile() wraps models and prepends '_orig_mod.' to all parameter
    names. This function removes that prefix so weights can be loaded into
    non-compiled models.
    """
    new_sd = {}
    for k, v in state_dict.items():
        new_key = k.replace('_orig_mod.', '')
        new_sd[new_key] = v
    return new_sd


def modify_slowfast_for_flow(model, in_channels=2):
    """Modify SlowFast-R50 stem convolutions from 3→2 input channels for optical flow.
    
    Modifies the first Conv3D in both slow and fast pathway stems.
    Transfers first 2 channels of pretrained weights.
    """
    modified = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv3d) and module.in_channels == 3:
            new_conv = nn.Conv3d(
                in_channels,
                module.out_channels,
                kernel_size=module.kernel_size,
                stride=module.stride,
                padding=module.padding,
                bias=module.bias is not None
            )
            with torch.no_grad():
                if in_channels <= 3:
                    new_conv.weight[:, :in_channels] = module.weight[:, :in_channels]
                else:
                    new_conv.weight[:, :3] = module.weight
                    new_conv.weight[:, 3:] = 0
                if module.bias is not None:
                    new_conv.bias.copy_(module.bias)
            
            parts = name.split('.')
            parent = model
            for p in parts[:-1]:
                parent = getattr(parent, p)
            setattr(parent, parts[-1], new_conv)
            modified += 1
            if modified >= 2:  # Only modify the two pathway stems
                break
    return model


class FusionClassifier(nn.Module):
    """Late fusion classifier for dual SlowFast streams.
    Concatenates RGB (2304-dim) and Flow (2304-dim) features → Dropout → Linear(4608, 2)
    """
    def __init__(self, rgb_dim, flow_dim, num_classes, dropout_rate=0.5):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(rgb_dim + flow_dim, num_classes)

    def forward(self, rgb_features, flow_features):
        combined = torch.cat([rgb_features, flow_features], dim=1)
        combined = self.dropout(combined)
        return self.fc(combined)


def build_violence_models(checkpoint_path, device):
    """Build dual SlowFast-R50 models and fusion classifier, load trained weights.
    
    Architecture:
        - RGB stream: SlowFast-R50 (3ch input, proj head → Identity) → 2304-dim
        - Flow stream: SlowFast-R50 (2ch input, proj head → Identity) → 2304-dim
        - Fusion: Dropout(0.5) → Linear(4608, 2)
        - Classes: Fight=0, NonFight=1
    """
    # Build RGB stream
    model_rgb = torch.hub.load(
        'facebookresearch/pytorchvideo', 'slowfast_r50', pretrained=False
    )
    model_rgb.blocks[6].proj = nn.Identity()  # Remove classification head
    
    # Build Flow stream (2-channel input)
    model_flow = torch.hub.load(
        'facebookresearch/pytorchvideo', 'slowfast_r50', pretrained=False
    )
    model_flow = modify_slowfast_for_flow(model_flow, in_channels=2)
    model_flow.blocks[6].proj = nn.Identity()
    
    # Build fusion classifier
    fusion = FusionClassifier(2304, 2304, 2)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Strip '_orig_mod.' prefix if checkpoint was saved from torch.compile() model
    sd_rgb = checkpoint['model_rgb']
    sd_flow = checkpoint['model_flow']
    sd_fusion = checkpoint['fusion']
    
    if any(k.startswith('_orig_mod.') for k in sd_rgb.keys()):
        print('  Detected torch.compile() checkpoint — stripping _orig_mod. prefix...')
        sd_rgb = strip_compiled_prefix(sd_rgb)
        sd_flow = strip_compiled_prefix(sd_flow)
        sd_fusion = strip_compiled_prefix(sd_fusion)
    
    model_rgb.load_state_dict(sd_rgb)
    model_flow.load_state_dict(sd_flow)
    fusion.load_state_dict(sd_fusion)
    
    model_rgb = model_rgb.to(device).eval()
    model_flow = model_flow.to(device).eval()
    fusion = fusion.to(device).eval()
    
    print(f'  Loaded violence models from {checkpoint_path}')
    if 'val_acc' in checkpoint:
        print(f'  Checkpoint val_acc: {checkpoint["val_acc"]:.2f}%')
    total_params = sum(
        sum(p.numel() for p in m.parameters())
        for m in [model_rgb, model_flow, fusion]
    )
    print(f'  Total parameters (RGB+Flow+Fusion): {total_params:,}')
    
    return model_rgb, model_flow, fusion


print('Violence detector architecture defined (Dual SlowFast-R50 + FusionClassifier).')


# ## Part 5: Load All Models

# In[6]:


# ════════════════════════════════════════════════════════════════════
# Load all pretrained models
# ════════════════════════════════════════════════════════════════════

print('Loading Gender Classifier (Body model)...')
gender_model_body = build_gender_model(CONFIG['gender_body_model_path'], device)

print('\nLoading Gender Classifier (Face model)...')
gender_model_face = build_gender_model(CONFIG['gender_face_model_path'], device)

print('\nLoading Violence Detector (Dual SlowFast)...')
model_rgb, model_flow, fusion = build_violence_models(CONFIG['violence_model_path'], device)

print('\nLoading YOLOv5s for Person Detection...')
yolo_model = YOLO(CONFIG['yolo_model'])
print(f'  YOLO model loaded: {CONFIG["yolo_model"]}')

# Optional auto-calibration of fight/nonfight output index using training clips
if CONFIG.get('enable_label_calibration', False):
    if 'calibrate_label_mapping_from_dataset' in globals():
        print('\nCalibrating violence label mapping from dataset clips...')
        try:
            calibrate_label_mapping_from_dataset(
                root_dir=CONFIG.get('calibration_root', ''),
                max_samples_per_class=CONFIG.get('calibration_samples_per_class', 24)
            )
        except Exception as e:
            print(f'  Label calibration failed, keeping configured fight_index={CONFIG.get("fight_index", 0)} | {e}')
    else:
        print('\nCalibration function not defined yet (Part 6.2 not run).')
        print('  Run Part 6.2 and re-run this cell to enable fight_index auto-calibration.')

print('\n' + '='*60)
print('All models loaded successfully!')
print('='*60)

# Clear GPU cache after loading
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    mem_used = torch.cuda.memory_allocated() / 1024**3
    mem_total = torch.cuda.get_device_properties(0).total_mem / 1024**3 if hasattr(torch.cuda.get_device_properties(0), 'total_mem') else torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f'GPU Memory used: {mem_used:.2f} GB / {mem_total:.1f} GB')


# ## Part 6: Preprocessing & Transforms

# ### 6.1 Gender Preprocessing

# In[7]:


# ═══════════════════════════════════════════════════════════════════════
# Gender Model Preprocessing
# Body model: Resize(448) → Normalize
# Face model: UpperBodyCrop(40%) → Resize(448) → Normalize
# ═══════════════════════════════════════════════════════════════════════

class UpperBodyCrop:
    """Crop the top portion of an image to focus on face/upper body."""
    def __init__(self, crop_ratio=0.40):
        self.crop_ratio = crop_ratio

    def __call__(self, image):
        """Args: image (PIL Image). Returns: cropped PIL Image."""
        width, height = image.size
        crop_height = int(height * self.crop_ratio)
        return image.crop((0, 0, width, crop_height))


# Body model transforms (no cropping)
body_transform = transforms.Compose([
    transforms.Resize((CONFIG['gender_image_size'], CONFIG['gender_image_size'])),
    transforms.ToTensor(),
    transforms.Normalize(mean=CONFIG['gender_mean'], std=CONFIG['gender_std'])
])

# Face model transforms (upper body crop first)
face_transform = transforms.Compose([
    UpperBodyCrop(crop_ratio=CONFIG['face_crop_ratio']),
    transforms.Resize((CONFIG['gender_image_size'], CONFIG['gender_image_size'])),
    transforms.ToTensor(),
    transforms.Normalize(mean=CONFIG['gender_mean'], std=CONFIG['gender_std'])
])


def preprocess_for_gender(crop_bgr, transform):
    """Preprocess a person crop (BGR numpy) for gender classification.
    
    Args:
        crop_bgr: numpy array (H, W, 3) in BGR format
        transform: torchvision transform pipeline
    Returns:
        tensor: (1, 3, 448, 448) ready for model input
    """
    # Convert BGR → RGB → PIL
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(crop_rgb)
    
    # Apply transforms
    tensor = transform(pil_img).unsqueeze(0)  # Add batch dimension
    return tensor.to(device)


print(f'Gender preprocessing configured:')
print(f'  Body model: Resize({CONFIG["gender_image_size"]}) → Normalize')
print(f'  Face model: UpperBodyCrop({CONFIG["face_crop_ratio"]*100:.0f}%) → Resize({CONFIG["gender_image_size"]}) → Normalize')


# ### 6.2 Violence Detection Preprocessing

# In[8]:


# ═══════════════════════════════════════════════════════════════════════
# Violence Detection Preprocessing
# Optical flow via Farneback, training-aligned clip assembly for SlowFast
# ═══════════════════════════════════════════════════════════════════════

# Global class mapping (may be auto-calibrated after model loading)
VIOLENCE_FIGHT_INDEX = CONFIG.get('fight_index', 0)
VIOLENCE_NONFIGHT_INDEX = 1 - VIOLENCE_FIGHT_INDEX

def resolve_calibration_root():
    """Return the first existing calibration root from configured candidates."""
    candidates = []
    if CONFIG.get('calibration_root', ''):
        candidates.append(CONFIG['calibration_root'])
    candidates.extend(CONFIG.get('calibration_root_candidates', []))

    seen = set()
    for root in candidates:
        if root in seen:
            continue
        seen.add(root)
        if os.path.isdir(os.path.join(root, 'Fight')) and os.path.isdir(os.path.join(root, 'NonFight')):
            return root
    return None


def compute_optical_flow_fixed(prev_gray, curr_gray, clip_value=None):
    """Compute dense optical flow and encode it exactly like training clips.

    Returns:
        flow_encoded: (H, W, 2) float32 in [0, 255]
    """
    if clip_value is None:
        clip_value = CONFIG['flow_clip_value']

    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray,
        flow=None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0
    )

    flow = np.clip(flow, -clip_value, clip_value)
    flow_encoded = (flow + clip_value) * (255.0 / (2.0 * clip_value))
    return flow_encoded.astype(np.float32)


def prepare_slowfast_inputs(frame_buffer, flow_buffer):
    """Prepare SlowFast inputs using the last training-aligned clip."""
    clip_len = CONFIG['clip_len']
    slow_frames = CONFIG['slow_frames']
    fast_frames = CONFIG['fast_frames']

    if len(frame_buffer) < clip_len or len(flow_buffer) < clip_len:
        raise ValueError(f'Need at least {clip_len} aligned RGB and flow frames.')

    frame_clip = list(frame_buffer)[-clip_len:]
    flow_clip = list(flow_buffer)[-clip_len:]

    slow_idx = np.linspace(0, clip_len - 1, slow_frames).astype(int)
    fast_idx = np.linspace(0, clip_len - 1, fast_frames).astype(int)

    # ── RGB Stream ──
    rgb_frames = np.array(frame_clip, dtype=np.float32) / 255.0  # (T, H, W, 3)
    rgb_tensor = torch.from_numpy(rgb_frames).permute(0, 3, 1, 2)  # (T, 3, H, W)

    slow_rgb = rgb_tensor[slow_idx].permute(1, 0, 2, 3)  # (3, T_slow, H, W)
    fast_rgb = rgb_tensor[fast_idx].permute(1, 0, 2, 3)  # (3, T_fast, H, W)

    rgb_mean = torch.tensor(CONFIG['rgb_mean'], dtype=torch.float32).view(3, 1, 1, 1)
    rgb_std = torch.tensor(CONFIG['rgb_std'], dtype=torch.float32).view(3, 1, 1, 1)
    slow_rgb = (slow_rgb - rgb_mean) / rgb_std
    fast_rgb = (fast_rgb - rgb_mean) / rgb_std

    # ── Flow Stream ──
    flow_frames = np.array(flow_clip, dtype=np.float32) / 255.0  # (T, H, W, 2)
    flow_tensor = torch.from_numpy(flow_frames).permute(0, 3, 1, 2)  # (T, 2, H, W)

    slow_flow = flow_tensor[slow_idx].permute(1, 0, 2, 3)  # (2, T_slow, H, W)
    fast_flow = flow_tensor[fast_idx].permute(1, 0, 2, 3)  # (2, T_fast, H, W)

    flow_mean = torch.tensor(CONFIG['flow_mean'], dtype=torch.float32).view(2, 1, 1, 1)
    flow_std = torch.tensor(CONFIG['flow_std'], dtype=torch.float32).view(2, 1, 1, 1)
    slow_flow = (slow_flow - flow_mean) / flow_std
    fast_flow = (fast_flow - flow_mean) / flow_std

    rgb_input = [
        slow_rgb.unsqueeze(0).to(device),
        fast_rgb.unsqueeze(0).to(device)
    ]
    flow_input = [
        slow_flow.unsqueeze(0).to(device),
        fast_flow.unsqueeze(0).to(device)
    ]

    return rgb_input, flow_input


def calibrate_label_mapping_from_dataset(root_dir=None, max_samples_per_class=None):
    """Auto-detect Fight class index by probing known Fight/NonFight training clips.

    This prevents silent label-order mismatch when loading external checkpoints.
    """
    global VIOLENCE_FIGHT_INDEX, VIOLENCE_NONFIGHT_INDEX

    if max_samples_per_class is None:
        max_samples_per_class = CONFIG.get('calibration_samples_per_class', 24)

    if root_dir is None or not os.path.isdir(root_dir):
        root_dir = resolve_calibration_root()

    if root_dir is None:
        print('Label calibration skipped: no valid calibration root found (Fight/NonFight folders missing).')
        return VIOLENCE_FIGHT_INDEX

    fight_dir = os.path.join(root_dir, 'Fight')
    nonfight_dir = os.path.join(root_dir, 'NonFight')

    def _collect_probs(class_dir, limit):
        files = sorted([f for f in os.listdir(class_dir) if f.endswith('.npy')])[:limit]
        probs_list = []
        for fname in files:
            path = os.path.join(class_dir, fname)
            try:
                data = np.load(path).astype(np.float32)
                rgb = data[..., :3] / 255.0
                flow = data[..., 3:5] / 255.0

                rgb_tensor = torch.from_numpy(rgb).permute(0, 3, 1, 2)
                flow_tensor = torch.from_numpy(flow).permute(0, 3, 1, 2)

                T = rgb_tensor.shape[0]
                slow_idx = np.linspace(0, max(0, T - 1), CONFIG['slow_frames']).astype(int)
                fast_idx = np.linspace(0, max(0, T - 1), CONFIG['fast_frames']).astype(int)

                slow_rgb = rgb_tensor[slow_idx].permute(1, 0, 2, 3)
                fast_rgb = rgb_tensor[fast_idx].permute(1, 0, 2, 3)
                slow_flow = flow_tensor[slow_idx].permute(1, 0, 2, 3)
                fast_flow = flow_tensor[fast_idx].permute(1, 0, 2, 3)

                rgb_mean = torch.tensor(CONFIG['rgb_mean'], dtype=torch.float32).view(3, 1, 1, 1)
                rgb_std = torch.tensor(CONFIG['rgb_std'], dtype=torch.float32).view(3, 1, 1, 1)
                flow_mean = torch.tensor(CONFIG['flow_mean'], dtype=torch.float32).view(2, 1, 1, 1)
                flow_std = torch.tensor(CONFIG['flow_std'], dtype=torch.float32).view(2, 1, 1, 1)

                slow_rgb = ((slow_rgb - rgb_mean) / rgb_std).unsqueeze(0).to(device)
                fast_rgb = ((fast_rgb - rgb_mean) / rgb_std).unsqueeze(0).to(device)
                slow_flow = ((slow_flow - flow_mean) / flow_std).unsqueeze(0).to(device)
                fast_flow = ((fast_flow - flow_mean) / flow_std).unsqueeze(0).to(device)

                with torch.no_grad():
                    with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                        rgb_features = model_rgb([slow_rgb, fast_rgb])
                        flow_features = model_flow([slow_flow, fast_flow])
                        logits = fusion(rgb_features, flow_features)
                        probs = torch.softmax(logits.float(), dim=1).cpu().numpy()[0]
                probs_list.append(probs)
            except Exception:
                continue

        if len(probs_list) == 0:
            return None
        return np.mean(np.array(probs_list), axis=0)

    fight_mean = _collect_probs(fight_dir, max_samples_per_class)
    nonfight_mean = _collect_probs(nonfight_dir, max_samples_per_class)

    if fight_mean is None or nonfight_mean is None:
        print('Label calibration skipped: no valid .npy clips were read.')
        return VIOLENCE_FIGHT_INDEX

    # Fight index should be the output neuron with larger probability on Fight clips
    # and lower probability on NonFight clips.
    margin_idx0 = fight_mean[0] - nonfight_mean[0]
    margin_idx1 = fight_mean[1] - nonfight_mean[1]
    VIOLENCE_FIGHT_INDEX = 0 if margin_idx0 >= margin_idx1 else 1
    VIOLENCE_NONFIGHT_INDEX = 1 - VIOLENCE_FIGHT_INDEX

    CONFIG['fight_index'] = VIOLENCE_FIGHT_INDEX
    CONFIG['calibration_root'] = root_dir
    print('Violence label calibration complete:')
    print(f'  Calibration root: {root_dir}')
    print(f'  mean probs on Fight clips:    {fight_mean}')
    print(f'  mean probs on NonFight clips: {nonfight_mean}')
    print(f'  Selected fight_index={VIOLENCE_FIGHT_INDEX}, nonfight_index={VIOLENCE_NONFIGHT_INDEX}')
    return VIOLENCE_FIGHT_INDEX


print('Violence preprocessing configured:')
print(f'  Clip length: {CONFIG["clip_len"]} frames (training-aligned)')
print(f'  Frame buffer: {CONFIG["frame_buffer_size"]} frames')
print(f'  Slow pathway: {CONFIG["slow_frames"]} frames')
print(f'  Fast pathway: {CONFIG["fast_frames"]} frames (sampled from 16-frame clip)')
print(f'  Optical flow: Farneback + fixed clip to ±{CONFIG["flow_clip_value"]} px/frame → [0,255]')
print(f'  RGB norm: μ={CONFIG["rgb_mean"]}, σ={CONFIG["rgb_std"]}')
print(f'  Flow norm: μ={CONFIG["flow_mean"]}, σ={CONFIG["flow_std"]}')
print(f'  Initial fight_index={VIOLENCE_FIGHT_INDEX} (auto-calibration enabled={CONFIG.get("enable_label_calibration", False)})')

# If models are already loaded, calibrate now so later cells use the correct class index.
if CONFIG.get('enable_label_calibration', False) and all(name in globals() for name in ['model_rgb', 'model_flow', 'fusion']):
    print('\nRunning post-definition label calibration...')
    try:
        calibrate_label_mapping_from_dataset(
            root_dir=CONFIG.get('calibration_root', ''),
            max_samples_per_class=CONFIG.get('calibration_samples_per_class', 24)
        )
    except Exception as e:
        print(f'  Post-definition calibration failed, using fight_index={CONFIG.get("fight_index", 0)} | {e}')


# ## Part 7: Person Detection, Tracking & Visibility Assessment

# ### 7.1 Person Detection (YOLOv5s)

# In[9]:


# ═══════════════════════════════════════════════════════════════════════
# Stage 1: Person Detection using YOLOv5s
# ═══════════════════════════════════════════════════════════════════════

def detect_persons(frame, yolo_model):
    """Detect persons in frame using YOLOv5s.
    
    Args:
        frame: BGR numpy array (H, W, 3)
        yolo_model: Ultralytics YOLO model
    
    Returns:
        detections: list of dicts with 'bbox' (x1, y1, x2, y2) and 'confidence'
    """
    results = yolo_model(
        frame,
        conf=CONFIG['det_confidence'],
        classes=[0],  # class 0 = person in COCO
        verbose=False,
        max_det=CONFIG['max_detections']
    )
    
    detections = []
    for result in results:
        boxes = result.boxes
        if boxes is not None and len(boxes) > 0:
            for i in range(len(boxes)):
                bbox = boxes.xyxy[i].cpu().numpy().astype(int)  # (x1, y1, x2, y2)
                conf = float(boxes.conf[i].cpu())
                detections.append({
                    'bbox': tuple(bbox),
                    'confidence': conf
                })
    
    # Sort by confidence and keep top N
    detections.sort(key=lambda x: x['confidence'], reverse=True)
    return detections[:CONFIG['max_detections']]


print('Person detection function defined (YOLOv5s).')


# ### 7.2 IoU-Based Person Tracker

# In[10]:


# ═══════════════════════════════════════════════════════════════════════
# IoU-Based Person Tracking with Gender History and Violence Counting
# ═══════════════════════════════════════════════════════════════════════

def compute_iou(box1, box2):
    """Compute IoU between two bounding boxes (x1, y1, x2, y2)."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    
    return inter / (union + 1e-8)


class TrackedPerson:
    """Represents a tracked person with gender history and violence count."""
    _next_id = 0
    
    def __init__(self, bbox, confidence=0.0):
        self.id = TrackedPerson._next_id
        TrackedPerson._next_id += 1
        self.bbox = bbox  # (x1, y1, x2, y2)
        self.confidence = confidence
        self.frames_unseen = 0
        self.gender_history = deque(maxlen=CONFIG['gender_history_len'])  # P(male) values
        self.violence_count = 0
        self.last_violence_time = 0.0
        self.visibility = 'upper'  # 'full' or 'upper'
        self.smoothed_gender = 0.5  # 0=Female, 1=Male
        self.gender_label = 'Unknown'
    
    def update_bbox(self, bbox, confidence=0.0):
        self.bbox = bbox
        self.confidence = confidence
        self.frames_unseen = 0
    
    def update_gender(self, p_male):
        """Add gender prediction and compute temporal average."""
        self.gender_history.append(p_male)
        self.smoothed_gender = np.mean(list(self.gender_history))
        self.gender_label = 'Male' if self.smoothed_gender >= 0.5 else 'Female'
    
    def increment_violence(self, current_time):
        """Increment violence count (max once per second)."""
        if current_time - self.last_violence_time >= 1.0:
            self.violence_count += 1
            self.last_violence_time = current_time


class PersonTracker:
    """Greedy IoU-based person tracker."""
    
    def __init__(self):
        self.tracks = []  # List of TrackedPerson
    
    def update(self, detections):
        """Match detections to existing tracks, create/terminate as needed.
        
        Args:
            detections: list of dicts with 'bbox' and 'confidence'
        Returns:
            active_tracks: list of TrackedPerson currently active
        """
        if not detections:
            # Age out all tracks
            for track in self.tracks:
                track.frames_unseen += 1
            self.tracks = [t for t in self.tracks if t.frames_unseen <= CONFIG['track_age_out']]
            return self.tracks
        
        matched_tracks = set()
        matched_dets = set()
        
        # Greedy IoU matching
        for det_idx, det in enumerate(detections):
            best_iou = 0
            best_track_idx = -1
            
            for track_idx, track in enumerate(self.tracks):
                if track_idx in matched_tracks:
                    continue
                iou = compute_iou(det['bbox'], track.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_track_idx = track_idx
            
            if best_iou > CONFIG['iou_threshold'] and best_track_idx >= 0:
                self.tracks[best_track_idx].update_bbox(det['bbox'], det['confidence'])
                matched_tracks.add(best_track_idx)
                matched_dets.add(det_idx)
        
        # Create new tracks for unmatched detections
        for det_idx, det in enumerate(detections):
            if det_idx not in matched_dets:
                self.tracks.append(TrackedPerson(det['bbox'], det['confidence']))
        
        # Age unmatched tracks
        for track_idx, track in enumerate(self.tracks):
            if track_idx not in matched_tracks and track_idx < len(self.tracks):
                track.frames_unseen += 1
        
        # Remove aged-out tracks
        self.tracks = [t for t in self.tracks if t.frames_unseen <= CONFIG['track_age_out']]
        
        return self.tracks
    
    def get_active_tracks(self):
        """Get currently visible tracks (frames_unseen == 0)."""
        return [t for t in self.tracks if t.frames_unseen == 0]


print('Person tracker defined (greedy IoU matching).')


# ### 7.3 Visibility Assessment (MediaPipe Pose)

# In[11]:


# ═══════════════════════════════════════════════════════════════════════
# Visibility Assessment via MediaPipe Pose Landmarker (Tasks API)
# Determines if full body or only upper body is visible
# ═══════════════════════════════════════════════════════════════════════

import urllib.request

# Download MediaPipe Pose Landmarker model (lite variant for speed)
POSE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
POSE_MODEL_PATH = os.path.join(CONFIG['output_dir'], 'pose_landmarker_lite.task')

if not os.path.exists(POSE_MODEL_PATH):
    print('Downloading Pose Landmarker model...')
    urllib.request.urlretrieve(POSE_MODEL_URL, POSE_MODEL_PATH)
    print(f'  Downloaded to {POSE_MODEL_PATH}')
else:
    print(f'  Pose model already exists at {POSE_MODEL_PATH}')

# Lower body landmark indices in MediaPipe Pose (33 landmarks)
# Same numbering across all mediapipe versions
LOWER_BODY_LANDMARK_IDS = [23, 24, 25, 26, 27, 28]  # hips, knees, ankles

# Create persistent PoseLandmarker using the new Tasks API
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

_pose_options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=POSE_MODEL_PATH),
    running_mode=VisionRunningMode.IMAGE,
    min_pose_detection_confidence=CONFIG['pose_min_confidence'],
    num_poses=1
)
pose_landmarker = PoseLandmarker.create_from_options(_pose_options)
print('  PoseLandmarker initialized (Tasks API).')


def check_full_body_visible(crop_bgr):
    """Check if full body is visible using MediaPipe Pose Landmarker.
    
    Full body = at least 4 of 6 lower-body landmarks visible (>0.5 confidence).
    
    Args:
        crop_bgr: person crop in BGR format
    Returns:
        'full' if full body visible, 'upper' otherwise
    """
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=crop_rgb)
    result = pose_landmarker.detect(mp_image)
    
    if not result.pose_landmarks:
        return 'upper'  # No pose detected → assume upper only
    
    landmarks = result.pose_landmarks[0]  # First detected pose
    visible_count = sum(
        1 for lm_id in LOWER_BODY_LANDMARK_IDS
        if landmarks[lm_id].visibility > CONFIG['lower_body_visibility_threshold']
    )
    
    return 'full' if visible_count >= CONFIG['min_visible_lower_landmarks'] else 'upper'


print('Visibility assessment defined (MediaPipe Pose Landmarker).')
print(f'  Full body = ≥{CONFIG["min_visible_lower_landmarks"]} of 6 lower-body landmarks visible (>{CONFIG["lower_body_visibility_threshold"]})')


# ## Part 8: Gender Classification & Fusion

# In[12]:


# ═══════════════════════════════════════════════════════════════════════
# Stage 2: Visibility-Conditioned Dual-Model Gender Classification
#
# Fusion formula:
#   Full body visible: g = 0.6 * P_male(body) + 0.4 * P_male(face)
#   Upper only:        g = P_male(face)
#
# Temporal smoothing: mean of last K=5 predictions
# Female if smoothed_g < 0.5, Male otherwise
# ═══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def classify_gender(crop_bgr, track, visibility):
    """Classify gender using dual EVA-02 models with visibility-conditioned fusion.
    
    Args:
        crop_bgr: person bounding box crop (BGR numpy)
        track: TrackedPerson instance
        visibility: 'full' or 'upper'
    Returns:
        gender_label: 'Male' or 'Female'
        p_male: fused P(male) probability
    """
    # Always run face model (upper body crop applied in transform)
    face_input = preprocess_for_gender(crop_bgr, face_transform)
    with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
        face_output = gender_model_face(face_input)
    face_probs = torch.softmax(face_output, dim=1)
    p_male_face = face_probs[0, 1].item()  # Index 1 = Male
    
    if visibility == 'full':
        # Run body model on full crop
        body_input = preprocess_for_gender(crop_bgr, body_transform)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
            body_output = gender_model_body(body_input)
        body_probs = torch.softmax(body_output, dim=1)
        p_male_body = body_probs[0, 1].item()
        
        # Visibility-conditioned fusion: 0.6 * body + 0.4 * face
        p_male = CONFIG['body_weight_full'] * p_male_body + CONFIG['face_weight_full'] * p_male_face
    else:
        # Upper only → use face model exclusively
        p_male = p_male_face
    
    # Update track with temporal smoothing
    track.visibility = visibility
    track.update_gender(p_male)
    
    return track.gender_label, track.smoothed_gender


print('Gender classification with visibility-conditioned fusion defined.')
print(f'  Full body: {CONFIG["body_weight_full"]}×body + {CONFIG["face_weight_full"]}×face')
print(f'  Upper only: face model only')
print(f'  Temporal smoothing: last {CONFIG["gender_history_len"]} predictions')


# ## Part 9: Violence Detection

# In[13]:


# ═══════════════════════════════════════════════════════════════════════
# Stage 3: Dual SlowFast Violence Detection
#
# RGB + Optical Flow streams → late fusion → Fight/NonFight
# Temporal smoothing: Fight if ≥1 of last 5 predictions are Fight (conf > 0.35)
# ═══════════════════════════════════════════════════════════════════════

class ViolenceDetector:
    """Manages training-aligned clip buffering, flow computation, and inference."""
    
    def __init__(self):
        clip_len = CONFIG['clip_len']
        self.frame_buffer = deque(maxlen=CONFIG['frame_buffer_size'])  # aligned RGB clip buffer
        self.flow_buffer = deque(maxlen=CONFIG['frame_buffer_size'])   # aligned flow clip buffer
        self.prev_gray = None
        self.prediction_history = deque(maxlen=CONFIG['violence_temporal_window'])
        self.current_prediction = 'NonFight'
        self.current_confidence = 0.0
        self.smoothed_prediction = 'NonFight'
        self.frame_count = 0
        self.clip_len = clip_len
        self.ema_bbox = None
        self.ema_alpha = 0.4
    
    def get_group_bbox(self, frame_shape, active_tracks):
        if not active_tracks:
            return None
        min_x = min(t.bbox[0] for t in active_tracks)
        min_y = min(t.bbox[1] for t in active_tracks)
        max_x = max(t.bbox[2] for t in active_tracks)
        max_y = max(t.bbox[3] for t in active_tracks)
        h, w = frame_shape[:2]
        margin_x = int((max_x - min_x) * 0.25)
        margin_y = int((max_y - min_y) * 0.25)
        return (max(0, min_x - margin_x), max(0, min_y - margin_y),
                min(w, max_x + margin_x), min(h, max_y + margin_y))

    def add_frame(self, frame_bgr, active_tracks=None):
        current_bbox = self.get_group_bbox(frame_bgr.shape, active_tracks)
        if current_bbox is not None:
            if self.ema_bbox is None:
                self.ema_bbox = current_bbox
            else:
                self.ema_bbox = tuple(int(self.ema_alpha * c + (1 - self.ema_alpha) * p)
                                      for c, p in zip(current_bbox, self.ema_bbox))
        if self.ema_bbox is not None:
            x1, y1, x2, y2 = self.ema_bbox
            if x2 > x1 and y2 > y1:
                crop = frame_bgr[y1:y2, x1:x2]
            else:
                crop = frame_bgr
        else:
            crop = frame_bgr
            
        resized = cv2.resize(crop, (CONFIG['violence_input_size'], CONFIG['violence_input_size']))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        self.frame_buffer.append(rgb)

        if self.prev_gray is None:
            zero_flow = np.full(
                (CONFIG['violence_input_size'], CONFIG['violence_input_size'], 2),
                127.5,
                dtype=np.float32
            )
            self.flow_buffer.append(zero_flow)
        else:
            flow = compute_optical_flow_fixed(self.prev_gray, gray)
            self.flow_buffer.append(flow)

        self.prev_gray = gray
        self.frame_count += 1
    
    def is_ready(self):
        """Check if the aligned training-sized clip is available."""
        return (
            len(self.frame_buffer) >= self.clip_len and
            len(self.flow_buffer) >= self.clip_len
        )
    
    @torch.no_grad()
    def predict(self):
        """Run violence detection on the latest training-aligned clip."""
        if not self.is_ready():
            return self.current_prediction, self.current_confidence, self.smoothed_prediction

        frame_list = list(self.frame_buffer)
        flow_list = list(self.flow_buffer)
        rgb_input, flow_input = prepare_slowfast_inputs(frame_list, flow_list)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
            rgb_features = model_rgb(rgb_input)
            flow_features = model_flow(flow_input)
            outputs = fusion(rgb_features, flow_features)

        probs = torch.softmax(outputs.float(), dim=1)
        pred_class = probs.argmax(dim=1).item()

        fight_idx = CONFIG.get('fight_index', 0)
        self.current_prediction = 'Fight' if pred_class == fight_idx else 'NonFight'
        self.current_confidence = probs[0, fight_idx].item()  # P(Fight)

        self.prediction_history.append({
            'class': self.current_prediction,
            'confidence': self.current_confidence
        })

        # Confidence-only temporal smoothing: count frames where P(Fight) > threshold
        # regardless of whether Fight was the argmax class. This catches cases where
        # P(Fight) is elevated (e.g. 0.3-0.5) but not yet the dominant class.
        fight_count = sum(
            1 for p in self.prediction_history
            if p['confidence'] > CONFIG['violence_confidence_threshold']
        )
        self.smoothed_prediction = (
            'Fight' if fight_count >= CONFIG['violence_min_count'] else 'NonFight'
        )

        return self.current_prediction, self.current_confidence, self.smoothed_prediction


print('Violence detector pipeline defined.')
print(f'  Clip: last {CONFIG["clip_len"]} aligned RGB/flow frames')
print(f'  Run every {CONFIG["violence_run_interval"]} frames')
print(f'  fight_index={CONFIG.get("fight_index", 0)} (auto-calibrated if enabled)')
print(f'  Temporal smoothing: ≥{CONFIG["violence_min_count"]} of {CONFIG["violence_temporal_window"]} = Fight (conf > {CONFIG["violence_confidence_threshold"]})')


# ## Part 10: Grad-CAM Spatial Evidence Localization

# In[14]:


# ═══════════════════════════════════════════════════════════════════════
# Stage 4: Grad-CAM on RGB SlowFast slow pathway (blocks[5])
#
# Produces heatmap showing WHERE the fused model detects violence
# Extracts bounding boxes from high-activation regions
# ═══════════════════════════════════════════════════════════════════════

class SlowFastGradCAM:
    """Grad-CAM for the RGB SlowFast-R50 slow pathway using fused fight logits."""
    
    def __init__(self, model_rgb, model_flow, fusion_model, target_block_idx=5):
        self.model_rgb = model_rgb
        self.model_flow = model_flow
        self.fusion_model = fusion_model
        self.target_block = model_rgb.blocks[target_block_idx]
        self.activations = None
        self.gradients = None
        
        self.target_block.register_forward_hook(self._forward_hook)
        self.target_block.register_full_backward_hook(self._backward_hook)
    
    def _forward_hook(self, module, input, output):
        if isinstance(output, (list, tuple)):
            self.activations = output[0].detach()
        else:
            self.activations = output.detach()
    
    def _backward_hook(self, module, grad_input, grad_output):
        if isinstance(grad_output, (list, tuple)):
            self.gradients = grad_output[0].detach()
        else:
            self.gradients = grad_output.detach()
    
    def generate(self, rgb_input, flow_input, frame_shape, crop_bbox=None):
        """Generate Grad-CAM heatmap from the fused Fight logit.

        Args:
            rgb_input: [slow_rgb, fast_rgb] list of tensors
            flow_input: [slow_flow, fast_flow] list of tensors
            frame_shape: original frame shape for upsampling
        
        Returns:
            heatmap, evidence_bboxes, overlay
        """
        self.model_rgb.eval()
        self.model_flow.eval()
        self.fusion_model.eval()

        slow_rgb = rgb_input[0].clone().requires_grad_(True)
        fast_rgb = rgb_input[1].clone()
        slow_flow = flow_input[0].clone()
        fast_flow = flow_input[1].clone()

        self.model_rgb.zero_grad()
        self.model_flow.zero_grad()
        self.fusion_model.zero_grad()

        rgb_features = self.model_rgb([slow_rgb, fast_rgb])
        with torch.no_grad():
            flow_features = self.model_flow([slow_flow, fast_flow])
        logits = self.fusion_model(rgb_features, flow_features)

        # Use calibrated fight class index for Grad-CAM target
        fight_index = CONFIG.get('fight_index', 0)
        fight_score = logits[:, fight_index].sum()
        fight_score.backward(retain_graph=False)

        if self.activations is None or self.gradients is None:
            return None, [], None

        weights = self.gradients.mean(dim=[2, 3, 4], keepdim=True)  # (B, C, 1, 1, 1)
        cam = (weights * self.activations).sum(dim=1)  # (B, T, H, W)
        cam = cam.mean(dim=1)  # (B, H, W)
        cam = F.relu(cam)

        cam = cam[0].detach().cpu().numpy()
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
            # Sharpen: power-law emphasizes peaks, suppresses low activations
            cam = cam ** 2.0
            # Re-normalize after sharpening
            cam = cam / (cam.max() + 1e-8)
        else:
            cam = np.zeros_like(cam)

        H, W = frame_shape[:2]
        if crop_bbox is not None:
            x1, y1, x2, y2 = crop_bbox
            crop_w = max(1, x2 - x1)
            crop_h = max(1, y2 - y1)
            crop_heatmap = cv2.resize(cam, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)
            heatmap = np.zeros((H, W), dtype=cam.dtype)
            x1_safe, y1_safe = max(0, x1), max(0, y1)
            x2_safe, y2_safe = min(W, x2), min(H, y2)
            h_safe = y2_safe - y1_safe
            w_safe = x2_safe - x1_safe
            if h_safe > 0 and w_safe > 0:
                heatmap[y1_safe:y2_safe, x1_safe:x2_safe] = crop_heatmap[:h_safe, :w_safe]
        else:
            heatmap = cv2.resize(cam, (W, H), interpolation=cv2.INTER_LINEAR)
        evidence_bboxes = self._extract_bboxes(heatmap)

        heatmap_uint8 = (heatmap * 255).astype(np.uint8)
        overlay_raw = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        # Mask: only show heatmap where activation is significant (> 0.15)
        # This prevents the uniform blue tint over the entire frame
        mask = (heatmap > 0.15).astype(np.float32)
        mask_3ch = np.stack([mask]*3, axis=-1)
        overlay = (overlay_raw * mask_3ch).astype(np.uint8)

        self.activations = None
        self.gradients = None

        return heatmap, evidence_bboxes, overlay
    
    def _extract_bboxes(self, heatmap):
        """Extract bounding boxes from high-activation regions."""
        threshold = CONFIG['gradcam_threshold'] * heatmap.max()
        mask = (heatmap >= threshold).astype(np.uint8) * 255

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        bboxes = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= CONFIG['gradcam_min_area']:
                x, y, w, h = cv2.boundingRect(contour)
                bboxes.append((x, y, w, h))

        return bboxes


# Initialize Grad-CAM on the fused violence logit
gradcam = SlowFastGradCAM(model_rgb, model_flow, fusion, target_block_idx=4)

print('Grad-CAM initialized on RGB SlowFast blocks[5] (slow pathway).')
print(f'  Backprop target: fused Fight logit from FusionClassifier (fight_index={CONFIG.get("fight_index", 0)}, block=4 for sharper maps)')
print(f'  Heatmap threshold: {CONFIG["gradcam_threshold"]} × max')
print(f'  Min contour area: {CONFIG["gradcam_min_area"]} px')
print(f'  Overlay alpha: {CONFIG["gradcam_alpha"]}')


# ## Part 11: Alert System & Evidence Capture

# In[15]:


# ═══════════════════════════════════════════════════════════════════════
# Stage 5: Rule-Based Alert Generation & Evidence Capture
#
# 1. Lone Woman at Night: 1 female + 0 males + nighttime + cooldown
# 2. Violence Against Women: female ∩ Grad-CAM bbox + violence_count ≥ 5 + cooldown
# 3. Evidence: raw snapshot + Grad-CAM overlay + 5s video clip + JSON metadata
# ═══════════════════════════════════════════════════════════════════════

class AlertSystem:
    """Rule-based alert generation with evidence capture."""
    
    def __init__(self):
        self.last_lone_woman_alert = 0.0
        self.last_violence_alert = 0.0
        self.last_evidence_capture = 0.0
        self.evidence_count = 0
        self.raw_frame_buffer = deque(maxlen=int(CONFIG['evidence_clip_duration'] * 30))  # 5s at 30fps
        self.alert_log = []
    
    def is_nighttime(self):
        """Check if current time is within night hours (22:00 - 06:00)."""
        hour = datetime.now().hour
        return hour >= CONFIG['night_start'] or hour < CONFIG['night_end']
    
    def add_raw_frame(self, frame):
        """Add raw frame to evidence buffer."""
        self.raw_frame_buffer.append(frame.copy())
    
    def check_lone_woman_alert(self, active_tracks, current_time):
        """Check lone woman at night alert.
        
        Triggered when: exactly 1 female, 0 males, nighttime, cooldown elapsed.
        """
        if not self.is_nighttime():
            return False, None
        
        females = [t for t in active_tracks if t.gender_label == 'Female']
        males = [t for t in active_tracks if t.gender_label == 'Male']
        
        if len(females) == 1 and len(males) == 0:
            if current_time - self.last_lone_woman_alert >= CONFIG['lone_woman_cooldown']:
                self.last_lone_woman_alert = current_time
                alert_info = {
                    'type': 'LONE_WOMAN_AT_NIGHT',
                    'timestamp': datetime.now().isoformat(),
                    'track_id': int(females[0].id),
                    'bbox': [int(v) for v in females[0].bbox]
                }
                self.alert_log.append(alert_info)
                return True, alert_info
        
        return False, None
    
    def check_violence_against_women(self, active_tracks, evidence_bboxes, 
                                      violence_prediction, current_time):
        """Check violence against women alert.
        
        Triggered when: female track overlaps Grad-CAM evidence bbox,
        violence_count >= 5, cooldown elapsed.
        """
        if violence_prediction != 'Fight' or not evidence_bboxes:
            return False, None
        
        females = [t for t in active_tracks if t.gender_label == 'Female']
        
        for track in females:
            tx1, ty1, tx2, ty2 = track.bbox
            
            # Check overlap with any evidence bbox
            for (ex, ey, ew, eh) in evidence_bboxes:
                ex2, ey2 = ex + ew, ey + eh
                
                # Check if bboxes overlap (IoU > 0)
                overlap_x = max(0, min(tx2, ex2) - max(tx1, ex))
                overlap_y = max(0, min(ty2, ey2) - max(ty1, ey))
                
                if overlap_x > 0 and overlap_y > 0:
                    # Increment violence count for this track
                    track.increment_violence(current_time)
                    
                    # Check persistence threshold
                    if track.violence_count >= CONFIG['violence_persistence_threshold']:
                        if current_time - self.last_violence_alert >= CONFIG['violence_alert_cooldown']:
                            self.last_violence_alert = current_time
                            alert_info = {
                                'type': 'VIOLENCE_AGAINST_WOMEN',
                                'timestamp': datetime.now().isoformat(),
                                'track_id': int(track.id),
                                'bbox': [int(v) for v in track.bbox],
                                'violence_count': int(track.violence_count),
                                'evidence_bboxes': evidence_bboxes
                            }
                            self.alert_log.append(alert_info)
                            return True, alert_info
        
        return False, None
    
    def capture_evidence(self, frame, heatmap, overlay, evidence_bboxes,
                          active_tracks, violence_confidence, current_time, fps):
        """Capture evidence package when violence confidence is high.
        
        Triggered when: violence_confidence >= 0.70 and cooldown (15s) elapsed.
        Captures: raw snapshot, Grad-CAM overlay, 5s video clip, JSON metadata.
        """
        if violence_confidence < CONFIG['evidence_confidence']:
            return False
        
        if current_time - self.last_evidence_capture < CONFIG['evidence_cooldown']:
            return False
        
        self.last_evidence_capture = current_time
        self.evidence_count += 1
        
        evidence_id = f'evidence_{self.evidence_count:04d}'
        evidence_path = os.path.join(CONFIG['evidence_dir'], evidence_id)
        os.makedirs(evidence_path, exist_ok=True)
        
        # 1. Raw snapshot
        cv2.imwrite(os.path.join(evidence_path, 'raw_snapshot.jpg'), frame)
        
        # 2. Grad-CAM overlay
        if overlay is not None:
            alpha = CONFIG['gradcam_alpha']
            blended = cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)
            # Draw evidence bboxes on overlay
            for (x, y, w, h) in evidence_bboxes:
                cv2.rectangle(blended, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.imwrite(os.path.join(evidence_path, 'gradcam_overlay.jpg'), blended)
        
        # 3. Video clip (async, last 5 seconds)
        if len(self.raw_frame_buffer) > 0:
            clip_path = os.path.join(evidence_path, 'clip.mp4')
            frames_for_clip = list(self.raw_frame_buffer)
            clip_thread = threading.Thread(
                target=self._write_clip, 
                args=(frames_for_clip, clip_path, fps)
            )
            clip_thread.start()
        
        # 4. JSON metadata
        females = [t for t in active_tracks if t.gender_label == 'Female']
        males = [t for t in active_tracks if t.gender_label == 'Male']
        women_in_evidence = 0
        for track in females:
            tx1, ty1, tx2, ty2 = track.bbox
            for (ex, ey, ew, eh) in evidence_bboxes:
                if (max(0, min(tx2, ex+ew) - max(tx1, ex)) > 0 and
                    max(0, min(ty2, ey+eh) - max(ty1, ey)) > 0):
                    women_in_evidence += 1
                    break
        
        metadata = {
            'evidence_id': evidence_id,
            'timestamp': datetime.now().isoformat(),
            'violence_confidence': float(violence_confidence),
            'num_females': len(females),
            'num_males': len(males),
            'women_in_evidence_region': int(women_in_evidence),
            'evidence_bboxes': [[int(v) for v in b] for b in evidence_bboxes],
            'track_details': [
                {
                    'id': int(t.id),
                    'gender': t.gender_label,
                    'bbox': [int(v) for v in t.bbox],
                    'violence_count': int(t.violence_count)
                } for t in active_tracks
            ]
        }
        with open(os.path.join(evidence_path, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f'  📁 Evidence captured: {evidence_path}')
        return True
    
    def _write_clip(self, frames, output_path, fps):
        """Write video clip asynchronously."""
        if not frames:
            return
        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        for f in frames:
            writer.write(f)
        writer.release()


print('Alert system defined:')
print(f'  Lone Woman at Night: 1F + 0M + night ({CONFIG["night_start"]}:00-{CONFIG["night_end"]}:00) + {CONFIG["lone_woman_cooldown"]}s cooldown')
print(f'  Violence Against Women: female ∩ Grad-CAM + violence_count ≥ {CONFIG["violence_persistence_threshold"]} + {CONFIG["violence_alert_cooldown"]}s cooldown')
print(f'  Evidence: conf ≥ {CONFIG["evidence_confidence"]} + {CONFIG["evidence_cooldown"]}s cooldown → snapshot + overlay + {CONFIG["evidence_clip_duration"]}s clip + JSON')


# ## Part 12: Frame Annotation & Visualization

# In[16]:


# ═══════════════════════════════════════════════════════════════════════
# Frame Annotation — draws all overlays on the output frame
# ═══════════════════════════════════════════════════════════════════════

def annotate_frame(frame, active_tracks, violence_pred, violence_conf,
                    smoothed_violence, heatmap, evidence_bboxes, overlay,
                    lone_woman_alert, violence_alert, fps, frame_num):
    """Draw all annotations on the frame.
    
    - Person bboxes colored by gender (pink=F, blue=M)
    - Gender labels with confidence
    - Violence status banner
    - Grad-CAM heatmap overlay (when violence detected)
    - Evidence bounding boxes (red)
    - Alert banners
    - Stats overlay (counts, FPS)
    """
    annotated = frame.copy()
    H, W = annotated.shape[:2]
    
    # ── Grad-CAM overlay (behind everything else) ──
    if overlay is not None and smoothed_violence == 'Fight':
        alpha = CONFIG['gradcam_alpha']
        annotated = cv2.addWeighted(annotated, 1 - alpha, overlay, alpha, 0)
    
    # ── Violence region bounding boxes ──
    if smoothed_violence == 'Fight':
        # Draw evidence contour bboxes if available
        if evidence_bboxes:
            for (x, y, w, h) in evidence_bboxes:
                cv2.rectangle(annotated, (x, y), (x+w, y+h), (0, 0, 255), 3)
                cv2.putText(annotated, 'VIOLENCE ZONE', (x, y-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Draw thick red bounding boxes around all tracked persons
        # to indicate they are in a violence region
        for track in active_tracks:
            x1, y1, x2, y2 = track.bbox
            # Thick red border signals violence involvement
            cv2.rectangle(annotated, (x1-2, y1-2), (x2+2, y2+2), (0, 0, 255), 3)
            # Violence confidence label
            viol_label = 'VIOLENCE'
            (tw2, th2), _ = cv2.getTextSize(viol_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(annotated, (x1, y2+2), (x1+tw2+4, y2+th2+8), (0, 0, 255), -1)
            cv2.putText(annotated, viol_label, (x1+2, y2+th2+4),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # ── Person bounding boxes ──
    for track in active_tracks:
        x1, y1, x2, y2 = track.bbox
        
        # Color by gender
        if track.gender_label == 'Female':
            color = (180, 105, 255)  # Pink (BGR)
        elif track.gender_label == 'Male':
            color = (255, 180, 0)    # Blue (BGR)
        else:
            color = (200, 200, 200)  # Gray
        
        # Draw bbox
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        
        # Gender label with confidence
        gender_conf = abs(track.smoothed_gender - 0.5) * 2  # 0-1 confidence
        label = f'ID:{track.id} {track.gender_label} ({gender_conf:.0%})'
        if track.visibility == 'upper':
            label += ' [upper]'
        
        # Label background
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (x1, y1-th-8), (x1+tw+4, y1), color, -1)
        cv2.putText(annotated, label, (x1+2, y1-5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Violence count indicator for females
        if track.gender_label == 'Female' and track.violence_count > 0:
            viol_label = f'V:{track.violence_count}'
            cv2.putText(annotated, viol_label, (x1, y2+15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # ── Violence status banner (top center) ──
    if smoothed_violence == 'Fight':
        banner_color = (0, 0, 200)  # Red
        banner_text = 'VIOLENCE DETECTED'
    else:
        banner_color = (0, 150, 0)  # Green
        banner_text = 'NO VIOLENCE'
    
    (tw, th), _ = cv2.getTextSize(banner_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    bx = (W - tw) // 2 - 10
    cv2.rectangle(annotated, (bx, 5), (bx + tw + 20, th + 20), banner_color, -1)
    cv2.putText(annotated, banner_text, (bx + 10, th + 12),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # ── Alert banners ──
    alert_y = th + 40
    if lone_woman_alert:
        alert_text = '⚠ LONE WOMAN AT NIGHT ALERT'
        cv2.rectangle(annotated, (10, alert_y), (W-10, alert_y+35), (0, 165, 255), -1)
        cv2.putText(annotated, alert_text, (20, alert_y+25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        alert_y += 40
    
    if violence_alert:
        alert_text = '⚠ VIOLENCE AGAINST WOMEN ALERT'
        cv2.rectangle(annotated, (10, alert_y), (W-10, alert_y+35), (0, 0, 255), -1)
        cv2.putText(annotated, alert_text, (20, alert_y+25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        alert_y += 40
    
    # ── Stats overlay (bottom left) ──
    females = sum(1 for t in active_tracks if t.gender_label == 'Female')
    males = sum(1 for t in active_tracks if t.gender_label == 'Male')
    unknown = len(active_tracks) - females - males
    
    stats = [
        f'Frame: {frame_num} | FPS: {fps:.1f}',
        f'Persons: {len(active_tracks)} (F:{females} M:{males})',
        f'Violence: {smoothed_violence} ({violence_conf:.2f})'
    ]
    
    for i, text in enumerate(stats):
        y_pos = H - 10 - (len(stats) - 1 - i) * 22
        cv2.putText(annotated, text, (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    return annotated


print('Frame annotation function defined.')


# ## Part 13: Main Processing Pipeline

# In[17]:


# ═══════════════════════════════════════════════════════════════════════
# MAIN PIPELINE: Video → Detection → Tracking → Gender → Violence
#                → Grad-CAM → Alerts → Annotated Output
# ═══════════════════════════════════════════════════════════════════════

def process_video(input_path, output_path):
    """Process a video through the complete 5-stage women safety pipeline.
    
    Pipeline per frame:
      1. YOLO person detection
      2. IoU tracking + MediaPipe visibility assessment
      3. Dual EVA-02 gender classification (body + face, visibility-conditioned fusion)
      4. Dual SlowFast violence detection (RGB + optical flow) — every N frames
      5. Grad-CAM evidence localization (when violence detected)
      6. Alert generation (lone woman, violence against women)
      7. Evidence capture (high confidence violence)
      8. Frame annotation & output
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open video: {input_path}')
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f'Input video: {input_path}')
    print(f'  Resolution: {width}x{height}, FPS: {fps:.1f}, Frames: {total_frames}')
    print(f'  Duration: {total_frames/fps:.1f} seconds')
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    tracker = PersonTracker()
    violence_detector = ViolenceDetector()
    alert_system = AlertSystem()
    
    frame_num = 0
    start_time = time.time()
    current_heatmap = None
    current_evidence_bboxes = []
    current_overlay = None
    violence_pred = 'NonFight'
    violence_conf = 0.0
    smoothed_violence = 'NonFight'
    lone_woman_alert_active = False
    violence_alert_active = False
    
    stats = {
        'total_frames': 0,
        'total_persons_detected': 0,
        'total_females': 0,
        'total_males': 0,
        'violence_frames': 0,
        'lone_woman_alerts': 0,
        'violence_alerts': 0,
        'evidence_captured': 0,
    }
    
    print(f'\nProcessing...')
    print('='*60)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_num += 1
            current_time = frame_num / fps  # Simulated time based on video position
            elapsed = time.time() - start_time
            processing_fps = frame_num / max(elapsed, 1e-8)
            
            # ── Stage 1: Person Detection ──
            detections = detect_persons(frame, yolo_model)
            
            # ── Stage 1b: IoU Tracking ──
            tracker.update(detections)
            active_tracks = tracker.get_active_tracks()
            stats['total_persons_detected'] += len(active_tracks)
            
            # ── Stage 2: Gender Classification ──
            for track in active_tracks:
                x1, y1, x2, y2 = track.bbox
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width, x2), min(height, y2)
                
                if x2 - x1 < 10 or y2 - y1 < 10:
                    continue
                
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                
                visibility = check_full_body_visible(crop)
                gender_label, p_male = classify_gender(crop, track, visibility)
                
                if gender_label == 'Female':
                    stats['total_females'] += 1
                elif gender_label == 'Male':
                    stats['total_males'] += 1
            
            # ── Stage 3: Violence Detection ──
            violence_detector.add_frame(frame, active_tracks)
            alert_system.add_raw_frame(frame)
            
            if (frame_num % CONFIG['violence_run_interval'] == 0 and
                violence_detector.is_ready()):
                
                violence_pred, violence_conf, smoothed_violence = violence_detector.predict()
                
                if smoothed_violence == 'Fight':
                    stats['violence_frames'] += 1
                    
                    if violence_conf > CONFIG['violence_confidence_threshold']:
                        try:
                            frame_list = list(violence_detector.frame_buffer)
                            flow_list = list(violence_detector.flow_buffer)
                            rgb_input, flow_input = prepare_slowfast_inputs(frame_list, flow_list)
                            
                            current_heatmap, current_evidence_bboxes, current_overlay = (
                                gradcam.generate(rgb_input, flow_input, frame.shape, crop_bbox=violence_detector.ema_bbox)
                            )
                        except Exception as e:
                            print(f'  Grad-CAM error at frame {frame_num}: {e}')
                            current_heatmap = None
                            current_evidence_bboxes = []
                            current_overlay = None
                else:
                    current_heatmap = None
                    current_evidence_bboxes = []
                    current_overlay = None
            
            # ── Stage 5: Alert System ──
            lone_woman_alert_active, lone_info = alert_system.check_lone_woman_alert(
                active_tracks, current_time
            )
            if lone_woman_alert_active:
                stats['lone_woman_alerts'] += 1
                print(f'  ⚠ LONE WOMAN AT NIGHT ALERT at frame {frame_num}')
            
            violence_alert_active, viol_info = alert_system.check_violence_against_women(
                active_tracks, current_evidence_bboxes, smoothed_violence, current_time
            )
            if violence_alert_active:
                stats['violence_alerts'] += 1
                print(f'  ⚠ VIOLENCE AGAINST WOMEN ALERT at frame {frame_num}')
            
            if smoothed_violence == 'Fight':
                captured = alert_system.capture_evidence(
                    frame, current_heatmap, current_overlay, current_evidence_bboxes,
                    active_tracks, violence_conf, current_time, fps
                )
                if captured:
                    stats['evidence_captured'] += 1
            
            # ── Stage 6: Annotate & Write ──
            annotated = annotate_frame(
                frame, active_tracks, violence_pred, violence_conf,
                smoothed_violence, current_heatmap, current_evidence_bboxes,
                current_overlay, lone_woman_alert_active, violence_alert_active,
                processing_fps, frame_num
            )
            writer.write(annotated)
            
            if frame_num % 100 == 0 or frame_num == total_frames:
                pct = frame_num / total_frames * 100 if total_frames > 0 else 0
                print(f'  Frame {frame_num}/{total_frames} ({pct:.1f}%) | '
                      f'FPS: {processing_fps:.1f} | '
                      f'Persons: {len(active_tracks)} | '
                      f'Violence raw/smoothed: {violence_pred}/{smoothed_violence} ({violence_conf:.2f})')
            
            stats['total_frames'] = frame_num
    
    except KeyboardInterrupt:
        print('\nProcessing interrupted by user.')
    
    finally:
        cap.release()
        writer.release()
    
    total_time = time.time() - start_time
    print('\n' + '='*60)
    print('PROCESSING COMPLETE')
    print('='*60)
    print(f'  Output: {output_path}')
    print(f'  Total frames: {stats["total_frames"]}')
    print(f'  Processing time: {total_time:.1f}s ({stats["total_frames"]/total_time:.1f} FPS)')
    print(f'  Persons detected: {stats["total_persons_detected"]} instances')
    print(f'  Gender counts: {stats["total_females"]} female, {stats["total_males"]} male')
    print(f'  Violence frames: {stats["violence_frames"]}')
    print(f'  Lone woman alerts: {stats["lone_woman_alerts"]}')
    print(f'  Violence-against-women alerts: {stats["violence_alerts"]}')
    print(f'  Evidence packages: {stats["evidence_captured"]}')
    
    return stats, alert_system.alert_log


print('Main processing pipeline defined.')


# In[18]:


# ════════════════════════════════════════════════════════════════════
# Quick diagnostics for violence branch (run before full pipeline)
# ════════════════════════════════════════════════════════════════════

def quick_violence_diagnostics(video_path=None, frames_to_probe=64):
    """Probe model outputs and buffer statistics before full inference."""
    if video_path is None:
        video_path = CONFIG['input_video']

    print('--- Violence Diagnostics ---')
    print(f"Configured fight_index: {CONFIG.get('fight_index', 0)}")

    # 1) Optional dataset-based calibration health check
    calib_root = CONFIG.get('calibration_root', '')
    if os.path.isdir(calib_root):
        print(f'Calibration root found: {calib_root}')
    else:
        print('Calibration root not found in this environment (skipping dataset check).')

    # 2) Short online probe from input video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f'Cannot open probe video: {video_path}')
        return

    probe_detector = ViolenceDetector()
    collected = []
    read_frames = 0

    # For diagnostics, simulate a center-crop person detection since we
    # don't have YOLO running here. This gives a much better flow signal
    # than full-frame downscaling from 1280x720 -> 224x224.
    class _FakeTrk:
        def __init__(self, bbox): self.bbox = bbox

    while read_frames < frames_to_probe:
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]
        # Use center 60% of frame as pseudo-person region
        fake_tracks = [_FakeTrk((int(w*0.2), int(h*0.1), int(w*0.8), int(h*0.9)))]
        probe_detector.add_frame(frame, fake_tracks)
        read_frames += 1

        if probe_detector.is_ready() and (read_frames % CONFIG['violence_run_interval'] == 0):
            pred, conf, smoothed = probe_detector.predict()
            collected.append((read_frames, pred, conf, smoothed))

    cap.release()

    print(f'Frames read: {read_frames}')
    print(f'RGB buffer len: {len(probe_detector.frame_buffer)} | Flow buffer len: {len(probe_detector.flow_buffer)}')

    if len(probe_detector.flow_buffer) > 0:
        flow_arr = np.array(probe_detector.flow_buffer, dtype=np.float32)
        print(f'Flow encoded stats [0..255 expected] -> min={flow_arr.min():.2f}, max={flow_arr.max():.2f}, mean={flow_arr.mean():.2f}')

    if not collected:
        print('No predictions collected yet. Increase frames_to_probe.')
        return

    fight_probs = [c[2] for c in collected]
    print(f'Collected predictions: {len(collected)}')
    print(f'Fight prob min/mean/max: {np.min(fight_probs):.3f} / {np.mean(fight_probs):.3f} / {np.max(fight_probs):.3f}')
    print('Last 10 predictions (frame, raw, conf, smoothed):')
    for row in collected[-10:]:
        print(f'  {row[0]} -> {row[1]} ({row[2]:.3f}) | smoothed={row[3]}')


# Run quick diagnostics before long full-video processing
quick_violence_diagnostics(CONFIG['input_video'], frames_to_probe=200)


# In[19]:


# ════════════════════════════════════════════════════════════════════
# Violence heatmap overlay frame from a train/Fight .npy clip
# Dataset layout: RWF2000-OPT-RGB / npy / train / Fight / *.npy
# .npy shape: (T, H, W, 5)  channels: 0-2=RGB, 3-4=optical flow (uint8)
# ════════════════════════════════════════════════════════════════════

def resolve_fight_npy_root():
    """Resolve Fight folder from common dataset paths."""
    candidates = [
        '/kaggle/input/rwf2000-opt-rgb/npy/train/Fight',
        '/kaggle/input/RWF2000-OPT-RGB/npy/train/Fight',
        '/kaggle/input/rwf2000-opt-rgb/npy/Train/Fight',
        '/kaggle/input/RWF2000-OPT-RGB/npy/Train/Fight',
    ]
    for path in candidates:
        if os.path.isdir(path):
            return path
    return candidates[0]

FIGHT_NPY_ROOT = resolve_fight_npy_root()


def render_heatmap_from_npy(npy_path, output_image_path=None):
    """Load a Fight .npy clip, run Grad-CAM, and display one overlaid frame."""
    if output_image_path is None:
        name = Path(npy_path).stem
        output_image_path = os.path.join(CONFIG['output_dir'], f'{name}_violence_overlay.jpg')

    # ── Load & split channels ──────────────────────────────────────
    data = np.load(npy_path).astype(np.float32)       # (T, H, W, 5)
    T = data.shape[0]
    rgb_raw  = data[..., :3]                           # (T, H, W, 3)  values 0-255
    flow_raw = data[..., 3:5]                          # (T, H, W, 2)  values 0-255

    # Normalise exactly as training does
    rgb_norm  = rgb_raw  / 255.0
    flow_norm = flow_raw / 255.0

    rgb_tensor  = torch.from_numpy(rgb_norm ).permute(0, 3, 1, 2)   # (T, 3, H, W)
    flow_tensor = torch.from_numpy(flow_norm).permute(0, 3, 1, 2)   # (T, 2, H, W)

    slow_idx = np.linspace(0, T - 1, CONFIG['slow_frames']).astype(int)
    fast_idx = np.linspace(0, T - 1, CONFIG['fast_frames']).astype(int)

    slow_rgb  = rgb_tensor[slow_idx].permute(1, 0, 2, 3)   # (3, Ts, H, W)
    fast_rgb  = rgb_tensor[fast_idx].permute(1, 0, 2, 3)
    slow_flow = flow_tensor[slow_idx].permute(1, 0, 2, 3)  # (2, Ts, H, W)
    fast_flow = flow_tensor[fast_idx].permute(1, 0, 2, 3)

    rgb_mean  = torch.tensor(CONFIG['rgb_mean'],  dtype=torch.float32).view(3, 1, 1, 1)
    rgb_std   = torch.tensor(CONFIG['rgb_std'],   dtype=torch.float32).view(3, 1, 1, 1)
    flow_mean = torch.tensor(CONFIG['flow_mean'], dtype=torch.float32).view(2, 1, 1, 1)
    flow_std  = torch.tensor(CONFIG['flow_std'],  dtype=torch.float32).view(2, 1, 1, 1)

    slow_rgb  = ((slow_rgb  - rgb_mean)  / rgb_std ).unsqueeze(0).to(device)
    fast_rgb  = ((fast_rgb  - rgb_mean)  / rgb_std ).unsqueeze(0).to(device)
    slow_flow = ((slow_flow - flow_mean) / flow_std).unsqueeze(0).to(device)
    fast_flow = ((fast_flow - flow_mean) / flow_std).unsqueeze(0).to(device)

    # ── Quick inference ────────────────────────────────────────────
    # NOTE: pytorchvideo SlowFast mutates the input list in-place during
    # forward (x[pathway] = output), so we pass CLONED lists here and
    # rebuild fresh lists for Grad-CAM below.
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
            rgb_feat  = model_rgb( [slow_rgb.clone(),  fast_rgb.clone() ])
            flow_feat = model_flow([slow_flow.clone(), fast_flow.clone()])
            logits    = fusion(rgb_feat, flow_feat)
            probs     = torch.softmax(logits.float(), dim=1).cpu().numpy()[0]

    fight_idx  = CONFIG.get('fight_index', 0)
    p_fight    = float(probs[fight_idx])
    pred_label = 'Fight' if probs.argmax() == fight_idx else 'NonFight'
    print(f'Clip: {Path(npy_path).name}  →  {pred_label}  P(Fight)={p_fight:.3f}  probs={probs}')

    # ── Grad-CAM ───────────────────────────────────────────────────
    # Use the middle frame as the representative display frame
    mid_frame_idx = T // 2
    frame_rgb_uint8 = rgb_raw[mid_frame_idx].astype(np.uint8)       # (H, W, 3) RGB
    frame_bgr       = cv2.cvtColor(frame_rgb_uint8, cv2.COLOR_RGB2BGR)
    frame_shape     = frame_bgr.shape

    # Rebuild fresh input lists — original tensors are still intact, only
    # the list objects were mutated by the model's in-place pathway update.
    rgb_input_gradcam  = [slow_rgb,  fast_rgb ]
    flow_input_gradcam = [slow_flow, fast_flow]

    heatmap, evidence_bboxes, overlay = gradcam.generate(
        rgb_input_gradcam, flow_input_gradcam, frame_shape
    )

    # ── Build overlay image ────────────────────────────────────────
    if overlay is not None:
        alpha   = CONFIG.get('gradcam_alpha', 0.45)
        blended = cv2.addWeighted(frame_bgr, 1 - alpha, overlay, alpha, 0)
    else:
        blended = frame_bgr.copy()

    # Draw evidence bounding boxes
    for (x, y, w, h) in evidence_bboxes:
        cv2.rectangle(blended, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(blended, 'EVIDENCE', (x, max(y - 5, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

    # Status banner
    color  = (0, 0, 255) if pred_label == 'Fight' else (0, 180, 0)
    banner = f'{pred_label}  |  P(Fight): {p_fight:.3f}  |  Frame {mid_frame_idx}/{T-1}'
    H_img, W_img = blended.shape[:2]
    cv2.rectangle(blended, (10, 10), (min(W_img - 10, 1050), 52), (0, 0, 0), -1)
    cv2.putText(blended, banner, (18, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

    cv2.imwrite(output_image_path, blended)
    print(f'Overlay frame saved → {output_image_path}')

    # ── Show side-by-side: original | overlay ──────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle(f'Violence Heatmap Overlay — {Path(npy_path).name}', fontsize=14, fontweight='bold')

    axes[0].imshow(frame_rgb_uint8)
    axes[0].set_title('Original Frame (middle of clip)', fontsize=11)
    axes[0].axis('off')

    axes[1].imshow(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f'Grad-CAM Overlay  ({pred_label}, P(Fight)={p_fight:.3f})', fontsize=11)
    axes[1].axis('off')

    plt.tight_layout()
    save_path = output_image_path.replace('.jpg', '_comparison.jpg')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'Comparison figure saved → {save_path}')

    return output_image_path


# ── Pick first Fight .npy and render ──────────────────────────────
if not os.path.isdir(FIGHT_NPY_ROOT):
    print(f'Fight .npy folder not found: {FIGHT_NPY_ROOT}')
    print('Update FIGHT_NPY_ROOT and re-run.')
else:
    npy_files = sorted(f for f in os.listdir(FIGHT_NPY_ROOT) if f.endswith('.npy'))
    if not npy_files:
        print(f'No .npy files found in {FIGHT_NPY_ROOT}')
    else:
        chosen = os.path.join(FIGHT_NPY_ROOT, npy_files[0])
        print(f'Using: {chosen}')
        render_heatmap_from_npy(chosen)


# ## Part 14: Run the Pipeline

# In[20]:


# ════════════════════════════════════════════════════════════════════
# Execute the pipeline on the configured input video
# ════════════════════════════════════════════════════════════════════

print('Starting Women Safety Surveillance Pipeline...')
print(f'Input:  {CONFIG["input_video"]}')
print(f'Output: {CONFIG["output_video"]}')
print(f'Evidence: {CONFIG["evidence_dir"]}/')
print(f'Fight class index (used by detector): {CONFIG.get("fight_index", 0)}')
print()

stats, alert_log = process_video(
    CONFIG['input_video'],
    CONFIG['output_video']
)


# ## Part 15: Results Visualization

# In[ ]:


# ════════════════════════════════════════════════════════════════════
# Visualize sample frames from the annotated output video
# ════════════════════════════════════════════════════════════════════

def show_sample_frames(video_path, num_samples=6):
    """Display sample frames from the annotated output video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f'Cannot open {video_path}')
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        print('No frames in video.')
        cap.release()
        return
    
    # Sample frames evenly
    sample_indices = np.linspace(0, total_frames - 1, num_samples).astype(int)
    
    cols = min(3, num_samples)
    rows = (num_samples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 4*rows))
    axes = np.atleast_2d(axes)
    fig.suptitle('Sample Annotated Frames', fontsize=16, fontweight='bold')
    
    for i, frame_idx in enumerate(sample_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            row, col = i // cols, i % cols
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            axes[row, col].imshow(frame_rgb)
            axes[row, col].set_title(f'Frame {frame_idx}', fontsize=10)
            axes[row, col].axis('off')
    
    # Hide unused axes
    for i in range(num_samples, rows * cols):
        row, col = i // cols, i % cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG['evidence_dir'], 'sample_frames.png'), dpi=150, bbox_inches='tight')
    plt.show()
    cap.release()


# Show sample annotated frames
if os.path.exists(CONFIG['output_video']):
    show_sample_frames(CONFIG['output_video'])
else:
    print(f'Output video not found: {CONFIG["output_video"]}')


# In[ ]:


# ════════════════════════════════════════════════════════════════════
# Display evidence captures and alert log
# ════════════════════════════════════════════════════════════════════

# Show alert log
print('Alert Log')
print('='*60)
if alert_log:
    for i, alert in enumerate(alert_log):
        print(f'\n  Alert {i+1}:')
        for k, v in alert.items():
            print(f'    {k}: {v}')
else:
    print('  No alerts triggered.')

# Show evidence snapshots
evidence_dirs = sorted(Path(CONFIG['evidence_dir']).glob('evidence_*'))
if evidence_dirs:
    print(f'\n\nEvidence Captures: {len(evidence_dirs)}')
    print('='*60)
    
    # Show up to 4 evidence captures
    show_dirs = evidence_dirs[:4]
    fig, axes = plt.subplots(len(show_dirs), 2, figsize=(14, 5*len(show_dirs)))
    if len(show_dirs) == 1:
        axes = axes.reshape(1, -1)
    
    for i, edir in enumerate(show_dirs):
        # Raw snapshot
        raw_path = edir / 'raw_snapshot.jpg'
        if raw_path.exists():
            raw = cv2.imread(str(raw_path))
            axes[i, 0].imshow(cv2.cvtColor(raw, cv2.COLOR_BGR2RGB))
            axes[i, 0].set_title(f'{edir.name} - Raw', fontsize=10)
        axes[i, 0].axis('off')
        
        # Grad-CAM overlay
        overlay_path = edir / 'gradcam_overlay.jpg'
        if overlay_path.exists():
            overlay_img = cv2.imread(str(overlay_path))
            axes[i, 1].imshow(cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB))
            axes[i, 1].set_title(f'{edir.name} - Grad-CAM', fontsize=10)
        axes[i, 1].axis('off')
        
        # Print metadata
        meta_path = edir / 'metadata.json'
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            print(f'\n  {edir.name}:')
            print('    Violence: VIOLENCE DETECTED')
            print(f'    Females: {meta["num_females"]}, Males: {meta["num_males"]}')
            print(f'    Women in evidence region: {meta["women_in_evidence_region"]}')
    
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG['evidence_dir'], 'evidence_summary.png'), dpi=150, bbox_inches='tight')
    plt.show()
else:
    print('\nNo evidence captures.')


# In[ ]:


# ════════════════════════════════════════════════════════════════════
# Final Summary
# ════════════════════════════════════════════════════════════════════

print('\n' + '='*70)
print('   WOMEN SAFETY SURVEILLANCE SYSTEM — PROCESSING SUMMARY')
print('='*70)
print(f'''
  Pipeline Architecture:
    Stage 1: YOLOv5s Person Detection (conf > {CONFIG["det_confidence"]}, max {CONFIG["max_detections"]})
             + IoU Tracking (threshold = {CONFIG["iou_threshold"]}, age-out = {CONFIG["track_age_out"]} frames)
  
    Stage 2: Dual EVA-02 Large Gender Classification
             Body model: full crop → {CONFIG["gender_image_size"]}×{CONFIG["gender_image_size"]}
             Face model: top {CONFIG["face_crop_ratio"]*100:.0f}% crop → {CONFIG["gender_image_size"]}×{CONFIG["gender_image_size"]}
             Fusion: {CONFIG["body_weight_full"]}×body + {CONFIG["face_weight_full"]}×face (full) | face-only (upper)
             Temporal smoothing: {CONFIG["gender_history_len"]}-frame window
  
    Stage 3: Dual SlowFast-R50 Violence Detection
             RGB stream (3ch) + Optical Flow stream (2ch, Farneback)
             Slow: {CONFIG["slow_frames"]} frames | Fast: {CONFIG["fast_frames"]} frames | Buffer: {CONFIG["frame_buffer_size"]}
             Temporal smoothing: ≥{CONFIG["violence_min_count"]}/{CONFIG["violence_temporal_window"]} = Fight (conf > {CONFIG["violence_confidence_threshold"]})
  
    Stage 4: Grad-CAM Evidence Localization
             Target: RGB SlowFast blocks[5] (slow pathway)
             Threshold: {CONFIG["gradcam_threshold"]}×max | Min area: {CONFIG["gradcam_min_area"]}px
  
    Stage 5: Alert System
             Lone Woman: 1F + 0M + night + {CONFIG["lone_woman_cooldown"]}s cooldown
             Violence: female ∩ evidence + count ≥ {CONFIG["violence_persistence_threshold"]} + {CONFIG["violence_alert_cooldown"]}s cooldown
             Evidence: conf ≥ {CONFIG["evidence_confidence"]} + {CONFIG["evidence_cooldown"]}s cooldown

  Results:
    Frames processed:          {stats["total_frames"]}
    Person detections:         {stats["total_persons_detected"]}
    Female classifications:    {stats["total_females"]}
    Male classifications:      {stats["total_males"]}
    Violence frames:           {stats["violence_frames"]}
    Lone woman alerts:         {stats["lone_woman_alerts"]}
    Violence-against-women:    {stats["violence_alerts"]}
    Evidence packages:         {stats["evidence_captured"]}
''')
print('='*70)

