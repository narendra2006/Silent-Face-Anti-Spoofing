import os
import cv2
import torch
import re
import torch.nn.functional as F
from collections import OrderedDict
import mediapipe as mp

from src.model_lib.MiniFASNet import MiniFASNetV1SE, MiniFASNetV2
from src.data_io.transform import SDKTestTransform
from src.utility import parse_model_name

mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

MODEL_DICT = {
    'MiniFASNetV1SE': MiniFASNetV1SE, 
    'MiniFASNetV2': MiniFASNetV2
}

class AntiSpoofPredict(object):
    def __init__(self, device_id):
        self.device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')

    def _load_model(self, model_path):
        model_name = os.path.basename(model_path)
        h, w, m_type, _ = parse_model_name(model_name)
        
        # Membentuk kerangka model
        self.model = MODEL_DICT[m_type](conv6_kernel=(h//16, w//16)).to(self.device)
        
        # Load weights dan bersihkan kunci (keys)
        sd = torch.load(model_path, map_location=self.device)
        clean_sd = OrderedDict()
        
        for k, v in sd.items():
            # 1. Buang FTGenerator
            if 'FTGenerator' in k:
                continue
            
            # 2. Buang prefix module. dan model.
            new_k = k.replace('module.', '').replace('model.', '')
            
            # 3. Perbaiki format list dari Sequential conv_3.0 jadi conv_3.model.0
            new_k = re.sub(r'conv_(\d+)\.(\d+)\.', r'conv_\1.model.\2.', new_k)
            
            # 4. Perbaikan untuk SE module jika ada
            new_k = new_k.replace('se_fc1', 'se_module.fc1')
            new_k = new_k.replace('se_bn1', 'se_module.bn1')
            new_k = new_k.replace('se_fc2', 'se_module.fc2')
            new_k = new_k.replace('se_bn2', 'se_module.bn2')
            
            clean_sd[new_k] = v
            
        # Gunakan strict=False agar tidak crash jika ada sisa key minor seperti num_batches_tracked
        self.model.load_state_dict(clean_sd, strict=False)

    def predict(self, img, path):
        self._load_model(path)
        self.model.eval()
        
        # Preprocessing gambar
        img = SDKTestTransform()(img).unsqueeze(0).to(self.device)
        
        # Inferensi (Prediksi)
        with torch.no_grad():
            return F.softmax(self.model(img), dim=-1).cpu().numpy()

    def get_bbox(self, image_bgr):
        """
        Menggantikan fungsi MTCNN menggunakan MediaPipe. 
        Mengembalikan [x, y, width, height]
        """
        height, width, _ = image_bgr.shape
        
        # MediaPipe membutuhkan format RGB
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        results = face_detection.process(image_rgb)
        
        # Jika tidak ada wajah terdeteksi
        if not results.detections:
            return None
        
        # Ambil wajah pertama yang terdeteksi
        detection = results.detections[0]
        bboxC = detection.location_data.relative_bounding_box
        
        # Konversi persentase ke pixel
        x = int(bboxC.xmin * width)
        y = int(bboxC.ymin * height)
        w = int(bboxC.width * width)
        h = int(bboxC.height * height)
        
        # Padding agar dagu/dahi tidak terpotong ekstrem
        padding_x = int(w * 0.1)
        padding_y = int(h * 0.1)
        
        x = max(0, x - padding_x)
        y = max(0, y - padding_y)
        w = min(width - x, w + (padding_x * 2))
        h = min(height - y, h + (padding_y * 2))
        
        return [x, y, w, h]
