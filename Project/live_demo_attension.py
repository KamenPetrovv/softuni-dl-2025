import cv2
import torch
import torch.nn as nn
import numpy as np
import pyaudio
import librosa
import threading
import time
from collections import deque, defaultdict
from torchvision import transforms
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1

# ==========================================
# CROSS-ATTENTION SOTA MODEL
# ==========================================
class CrossAttentionActiveSpeakerModel(nn.Module):
    def __init__(self, dropout_audio=0.2, hidden_size=128):
        super(CrossAttentionActiveSpeakerModel, self).__init__()
        
        self.audio_conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1))
        )
        self.audio_dropout = nn.Dropout(dropout_audio)
        
        self.visual_proj = nn.Linear(512, hidden_size)
        self.audio_proj = nn.Linear(32, hidden_size)
        
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=4, batch_first=True)
        
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size, 
            num_layers=1, 
            batch_first=True
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, v, a):
        B, T, V_dim = v.size()
        _, _, C, H, W = a.size()
        
        a_flat = a.view(B * T, C, H, W) 
        
        # Audio extraction and standard dropout
        a_feat = self.audio_conv(a_flat).view(B * T, -1) 
        a_feat = self.audio_dropout(a_feat)
        a_feat = a_feat.view(B, T, -1) 
        
        v_proj = self.visual_proj(v) # Shape: (B, 15, 128)
        a_proj = self.audio_proj(a_feat) # Shape: (B, 15, 128)
        
        attn_out, _ = self.cross_attn(query=v_proj, key=a_proj, value=a_proj)
        gru_out, _ = self.gru(attn_out)
        final_feat = gru_out.mean(dim=1) 
        
        main_out = self.classifier(final_feat)
        
        if self.training:
            # Return the mean of the projections to apply Contrastive Loss
            return main_out, v_proj.mean(dim=1), a_proj.mean(dim=1)
            
        return main_out

# --- CONFIGURATION ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
WINDOW_SIZE = 15
SAMPLE_RATE = 16000
CHUNK = 1024 

print("Loading Face Models...")
mtcnn = MTCNN(keep_all=True, device=DEVICE) # keep_all=True is required for multi-face!
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)

print("Loading Cross-Attention Temporal Model...")
model = CrossAttentionActiveSpeakerModel(hidden_size=128).to(DEVICE)
model.load_state_dict(torch.load("../best_model.pth", map_location=DEVICE, weights_only=True))
model.eval()

v_transform = transforms.Compose([
    transforms.Resize((160, 160)), transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

audio_buffer = deque(maxlen=int(SAMPLE_RATE * 0.6))
audio_lock = threading.Lock()
active_faces = {}
faces_lock = threading.Lock()
latest_frame = None
frame_lock = threading.Lock()

def audio_callback(in_data, frame_count, time_info, status):
    audio_data = np.frombuffer(in_data, dtype=np.float32)
    with audio_lock:
        audio_buffer.extend(audio_data)
    return (in_data, pyaudio.paContinue)

p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paFloat32, channels=1, rate=SAMPLE_RATE, input=True, frames_per_buffer=CHUNK, stream_callback=audio_callback)
stream.start_stream()

def ai_worker():
    global latest_frame, active_faces
    
    # Dynamic dictionary to handle infinite IDs as people come and go
    person_buffers = defaultdict(lambda: deque(maxlen=WINDOW_SIZE))
    
    # Lightweight Centroid Tracker variables
    tracked_faces = {} # {id: (centroid_x, centroid_y)}
    next_id = 0
    max_distance = 100 # Maximum pixel distance a face can move between frames
    
    while True:
        time.sleep(1/30) 
        
        with frame_lock:
            if latest_frame is None: continue
            rgb_frame = latest_frame.copy()
            
        boxes, _ = mtcnn.detect(rgb_frame)
        current_detected_this_frame = {}
        
        if boxes is not None:
            # 1. LIGHTWEIGHT CENTROID TRACKER
            new_tracked = {}
            for box in boxes:
                cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                best_id = None
                best_dist = float('inf')
                
                # Find the closest existing tracked face
                for tid, (tcx, tcy) in tracked_faces.items():
                    dist = ((cx - tcx)**2 + (cy - tcy)**2)**0.5
                    if dist < max_distance and dist < best_dist:
                        best_dist = dist
                        best_id = tid
                
                if best_id is not None:
                    new_tracked[best_id] = (cx, cy)
                    current_detected_this_frame[best_id] = box
                    del tracked_faces[best_id] # Consume ID
                else:
                    new_tracked[next_id] = (cx, cy)
                    current_detected_this_frame[next_id] = box
                    next_id += 1
            
            tracked_faces = new_tracked # Update state for next frame
            
            # 2. FEATURE EXTRACTION
            for t_id, box in current_detected_this_frame.items():
                x1, y1, x2, y2 = [int(b) for b in box]
                h, w, _ = rgb_frame.shape
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
                
                face = rgb_frame[y1:y2, x1:x2]
                if face.size > 0:
                    face_pil = Image.fromarray(face)
                    t = v_transform(face_pil).unsqueeze(0).to(DEVICE)
                    with torch.no_grad():
                        face_tensor = resnet(t).cpu().squeeze(0)
                    person_buffers[t_id].append(face_tensor)
        
        # Run Audio Sync Math
        with audio_lock:
            current_audio = np.array(audio_buffer)
            
        if len(current_audio) >= SAMPLE_RATE * 0.5:
            target_audio = current_audio[-int(SAMPLE_RATE * 0.5):]
            step_size = len(target_audio) // WINDOW_SIZE
            raw_mels = []
            
            for i in range(WINDOW_SIZE):
                start = i * step_size
                clip = target_audio[start : start + step_size]
                if len(clip) < 512: clip = np.pad(clip, (0, max(0, 512 - len(clip))))
                mel = librosa.feature.melspectrogram(y=clip, sr=SAMPLE_RATE, n_mels=64, hop_length=128, n_fft=512)
                if mel.shape[1] < 5: mel = np.pad(mel, ((0,0), (0, 5-mel.shape[1])))
                raw_mels.append(mel[:, :5])
                
            window_max = np.max([np.max(m) for m in raw_mels])
            if window_max == 0: window_max = 1.0
            
            audio_micro_specs = []
            for mel in raw_mels:
                mel_db = (librosa.power_to_db(mel, ref=window_max) + 40) / 40
                audio_micro_specs.append(torch.tensor(mel_db).unsqueeze(0))
            a_tensor = torch.stack(audio_micro_specs).unsqueeze(0).to(DEVICE) # Shape: [1, 15, 1, 64, 5]
            
            # 3. MULTI-PERSON GPU BATCHING
            ready_ids = []
            v_tensors = []
            
            new_state = {}
            for t_id, box in current_detected_this_frame.items():
                # Keep the box on screen even if buffer isn't full yet (prevents flickering)
                old_prob = active_faces.get(t_id, (None, 0.0))[1]
                new_state[t_id] = (box, old_prob)
                
                # If buffer is full, queue for inference
                if len(person_buffers[t_id]) == WINDOW_SIZE:
                    ready_ids.append(t_id)
                    v_tensors.append(torch.stack(list(person_buffers[t_id])))
            
            if ready_ids:
                v_batch = torch.stack(v_tensors).to(DEVICE) # Shape: [N, 15, 512]
                a_batch = a_tensor.repeat(len(ready_ids), 1, 1, 1, 1) # Broadcast audio to all faces
                
                with torch.no_grad():
                    out = model(v_batch, a_batch)
                    probs = torch.softmax(out, dim=1)[:, 1].tolist() # Get 'Speaking' probability for everyone
                    
                # Update the state with actual predictions
                for idx, prob in zip(ready_ids, probs):
                    new_state[idx] = (current_detected_this_frame[idx], prob)
                    
            with faces_lock:
                active_faces = new_state

# Start background AI worker
ai_thread = threading.Thread(target=ai_worker, daemon=True)
ai_thread.start()

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
print("Starting Live Multi-Face Inference... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret: break

    with frame_lock:
        latest_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    with faces_lock:
        for face_id, (box, prob) in active_faces.items():
            x1, y1, x2, y2 = [int(b) for b in box]
            is_speaking = prob > 0.75 
            
            label = f"ID:{face_id} SPK ({prob:.2f})" if is_speaking else f"ID:{face_id} SIL ({prob:.2f})"
            color = (0, 255, 0) if is_speaking else (0, 0, 255)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow('Live Temporal Active Speaker', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
stream.stop_stream()
stream.close()
p.terminate()
cv2.destroyAllWindows()