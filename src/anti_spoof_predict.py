import os
import cv2
import torch
import re
import numpy as np
import torch.nn.functional as F
from collections import OrderedDict
from retinaface import RetinaFace
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
        self.model = None
        self._loaded_model_path = None
        self.mtcnn = MTCNN()
        self.haar = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        print(f"✅ AntiSpoofPredict siap di device: {self.device}")

    def _load_model(self, model_path):
        if self._loaded_model_path == model_path and self.model is not None:
            return

        model_name = os.path.basename(model_path)
        h, w, m_type, _ = parse_model_name(model_name)
        self.model = MODEL_DICT[m_type](conv6_kernel=(h // 16, w // 16)).to(self.device)

        sd = torch.load(model_path, map_location=self.device)
        clean_sd = OrderedDict()
        for k, v in sd.items():
            if 'FTGenerator' in k:
                continue
            new_k = k.replace('module.', '').replace('model.', '')
            new_k = re.sub(r'conv_(\d+)\.(\d+)\.', r'conv_\1.model.\2.', new_k)
            new_k = new_k.replace('se_fc1', 'se_module.fc1')
            new_k = new_k.replace('se_bn1', 'se_module.bn1')
            new_k = new_k.replace('se_fc2', 'se_module.fc2')
            new_k = new_k.replace('se_bn2', 'se_module.bn2')
            clean_sd[new_k] = v

        self.model.load_state_dict(clean_sd, strict=False)
        self._loaded_model_path = model_path
        print(f"✅ Model di-load: {model_name}")

    def _add_padding(self, img, x, y, w, h, ratio=0.2):
        """Tambah padding proporsional agar konteks wajah tidak terpotong"""
        h_img, w_img = img.shape[:2]
        pad_x = int(w * ratio)
        pad_y = int(h * ratio)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(w_img, x + w + pad_x)
        y2 = min(h_img, y + h + pad_y)
        return [x1, y1, x2 - x1, y2 - y1]

    def get_bbox(self, img):
        """
        Cascade Detector: RetinaFace → MTCNN → Haar → Full Frame
        Setiap level adalah fallback dari level sebelumnya.
        """
        h_img, w_img = img.shape[:2]
        full_frame = [0, 0, w_img, h_img]

        # ── LEVEL 1: RetinaFace (paling akurat, tahan foto miring) ──
        try:
            faces = RetinaFace.detect_faces(img)
            if isinstance(faces, dict) and len(faces) > 0:
                best = max(faces.values(), key=lambda f: f['score'])
                if best['score'] >= 0.90:
                    x1, y1, x2, y2 = best['facial_area']
                    return self._add_padding(img, x1, y1, x2 - x1, y2 - y1)
        except Exception as e:
            pass  # Lanjut ke level berikutnya

        # ── LEVEL 2: MTCNN (bagus untuk wajah kecil) ──
        try:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            faces = self.mtcnn.detect_faces(img_rgb)
            if faces and len(faces) > 0:
                best = max(faces, key=lambda f: f['confidence'])
                if best['confidence'] >= 0.85:
                    x, y, w, h = best['box']
                    return self._add_padding(img, max(0, x), max(0, y), w, h)
        except Exception as e:
            pass  # Lanjut ke level berikutnya

        # ── LEVEL 3: Haar Cascade (ringan, tidak butuh GPU) ──
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.haar.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
            if len(faces) > 0:
                # Ambil wajah terbesar = wajah paling relevan
                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                return self._add_padding(img, x, y, w, h)
        except Exception as e:
            pass  # Lanjut ke full frame

        # ── LEVEL 4: Full Frame (last resort) ──
        return full_frame

    def predict(self, img, path):
        """
        Menerima gambar yang SUDAH di-crop oleh CropImage.
        Tidak perlu resize manual karena CropImage sudah handle ini.
        """
        self._load_model(path)
        self.model.eval()

        img_tensor = SDKTestTransform()(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            return F.softmax(self.model(img_tensor), dim=-1).cpu().numpy()
