import os
import cv2
import torch
import re
import torch.nn.functional as F
from collections import OrderedDict
from mtcnn import MTCNN

from src.model_lib.MiniFASNet import MiniFASNetV1SE, MiniFASNetV2
from src.data_io.transform import SDKTestTransform
from src.utility import parse_model_name

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

    def get_bbox(self, img):
        # 1. Inisialisasi MTCNN
        detector = MTCNN()
        
        # 2. MTCNN butuh format RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        try:
            faces = detector.detect_faces(img_rgb)
            
            # 3. Jika wajah ditemukan secara normal
            if len(faces) > 0:
                box = faces[0]['box']
                x, y, w, h = box
                x = max(0, x)
                y = max(0, y)
                return [x, y, w, h]
            else:
                # Fallback 1: MTCNN jalan tapi mengembalikan list kosong
                return [0, 0, img.shape[1], img.shape[0]]
                
        except Exception as e:
            # Fallback 2: MTCNN CRASH (seperti error ValueError: shape (0, 48, 48, 3))
            # Sangat berguna untuk foto extreme close-up
            print("⚠️ [Sistem] MTCNN gagal memotong wajah. Menggunakan seluruh frame gambar.")
            return [0, 0, img.shape[1], img.shape[0]]
