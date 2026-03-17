import cv2
import numpy as np
from collections import deque

# --- Extracted functions/classes for isolated testing ---
CONFIG = {
    'clip_len': 32,
    'frame_buffer_size': 32,
    'violence_temporal_window': 30,
    'violence_input_size': 224
}

def compute_optical_flow_fixed(prev_frame, curr_frame):
    flow = cv2.calcOpticalFlowFarneback(
        prev_frame, curr_frame, None, 
        pyr_scale=0.5, levels=3, winsize=15, 
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    
    flow_x = flow[..., 0]
    flow_y = flow[..., 1]
    
    flow_x_norm = cv2.normalize(flow_x, None, 0, 255, cv2.NORM_MINMAX)
    flow_y_norm = cv2.normalize(flow_y, None, 0, 255, cv2.NORM_MINMAX)
    
    flow_mapped = np.stack([flow_x_norm, flow_y_norm], axis=-1)
    return flow_mapped.astype(np.float32)

class MockTrack:
    def __init__(self, bbox):
        self.bbox = bbox

class ViolenceDetector:
    def __init__(self):
        clip_len = CONFIG['clip_len']
        self.frame_buffer = deque(maxlen=CONFIG['frame_buffer_size']) 
        self.flow_buffer = deque(maxlen=CONFIG['frame_buffer_size'])  
        self.prev_gray = None
        self.prediction_history = deque(maxlen=CONFIG['violence_temporal_window'])
        self.current_prediction = 'NonFight'
        self.current_confidence = 0.0
        self.smoothed_prediction = 'NonFight'
        self.frame_count = 0
        self.clip_len = clip_len
        self.ema_bbox = None  # (x1, y1, x2, y2)
        self.ema_alpha = 0.1  # Smoothing factor
    
    def get_group_bbox(self, frame_shape, active_tracks):
        if not active_tracks:
            return None
        min_x = min(t.bbox[0] for t in active_tracks)
        min_y = min(t.bbox[1] for t in active_tracks)
        max_x = max(t.bbox[2] for t in active_tracks)
        max_y = max(t.bbox[3] for t in active_tracks)
        h, w = frame_shape[:2]
        margin_x = int((max_x - min_x) * 0.1)
        margin_y = int((max_y - min_y) * 0.1)
        return (max(0, min_x - margin_x), max(0, min_y - margin_y),
                min(w, max_x + margin_x), min(h, max_y + margin_y))
    
    def add_frame(self, frame_bgr, active_tracks=None):
        current_bbox = self.get_group_bbox(frame_bgr.shape, active_tracks)
        if current_bbox is not None:
            if self.ema_bbox is None:
                self.ema_bbox = current_bbox
            else:
                self.ema_bbox = tuple(
                    int(self.ema_alpha * c + (1 - self.ema_alpha) * p)
                    for c, p in zip(current_bbox, self.ema_bbox)
                )
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

# --- Test execution ---
def run_test():
    cap = cv2.VideoCapture('/home/vivek/Documents/women safety/women safety-simple/Bengaluru_News_Bengaluru_Woman_Assaulted_In_Broad_Daylight_CCTV_Captures_Attack_720P.mp4')
    if not cap.isOpened():
        print("Error: Could not open video")
        return
        
    vd = ViolenceDetector()
    frame_count = 0
    non_zero_flow_counts = []
    
    # We will simulate a bounding box for the people in the video
    # Assuming the video has people around the center
    # This simplifies testing without running YOLO
    dummy_tracks = [MockTrack([200, 200, 400, 500]), MockTrack([250, 250, 450, 550])]
    
    print("Testing patched ViolenceDetector optical flow computation...")
    while frame_count < 30: # test 30 frames
        ret, frame = cap.read()
        if not ret:
            break
            
        vd.add_frame(frame, dummy_tracks)
        
        # Check flow buffer
        # (The flow buffer gives us [H, W, 2] shaped flow values arrays)
        latest_flow = vd.flow_buffer[-1]
        
        # Optical flow values should be in range ~0-255 after cv2.normalize
        # Check if values are not exactly 127.5 (which indicates zero motion)
        deviation = np.abs(latest_flow - 127.0).mean() # using 127 since cv2.normalize may center closely around there
        non_zero_flow_counts.append(deviation)
        print(f"Frame {frame_count}: shape={latest_flow.shape}, EMA bbox={vd.ema_bbox}, average deviation from 127={deviation:.4f}")
        
        frame_count += 1
        
    cap.release()
    
    print("\n--- Summary ---")
    avg_deviation = sum(non_zero_flow_counts[1:]) / len(non_zero_flow_counts[1:])  # Skip frame 0
    print(f"Average optical flow deviation from center (non-zero motion): {avg_deviation:.4f}")
    if avg_deviation > 1.0:
        print("SUCCESS: Optical flow is successfully capturing motion!")
    else:
        print("FAILURE: Optical flow values are still mostly zero.")

if __name__ == "__main__":
    run_test()
