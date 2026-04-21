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
        self.model = None
        self._loaded_model_path = None  # ✅ Tracker agar model tidak reload berulang
        self.detector = MTCNN()         # ✅ Inisialisasi MTCNN sekali saja

    def _load_model(self, model_path):
        # ✅ Skip reload jika model yang sama sudah di-load sebelumnya
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
        self._loaded_model_path = model_path  # ✅ Simpan path model yang sudah di-load
        print(f"✅ Model berhasil di-load: {model_name}")

    def predict(self, img, path):
        self._load_model(path)
        self.model.eval()

        img_tensor = SDKTestTransform()(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            return F.softmax(self.model(img_tensor), dim=-1).cpu().numpy()

    def get_bbox(self, img):
        """
        Mendeteksi bounding box wajah menggunakan MTCNN.
        Jika wajah tidak ditemukan atau MTCNN gagal,
        fallback ke seluruh frame gambar secara otomatis.
        """
        # ✅ FIX: Indentasi sekarang konsisten dan benar
        full_frame = [0, 0, img.shape[1], img.shape[0]]

        # Konversi ke RGB karena MTCNN membutuhkannya
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        try:
            faces = self.detector.detect_faces(img_rgb)

            if len(faces) == 0:
                # Fallback 1: MTCNN berjalan tapi tidak menemukan wajah
                print("⚠️ [Info] Wajah tidak terdeteksi. Menggunakan seluruh frame.")
                return full_frame

            # ✅ Ambil wajah dengan confidence tertinggi (bukan sekadar faces[0])
            best_face = max(faces, key=lambda f: f['confidence'])
            
            # ✅ Tambahan: abaikan deteksi dengan confidence rendah
            if best_face['confidence'] < 0.85:
                print(f"⚠️ [Info] Confidence rendah ({best_face['confidence']:.2f}). Menggunakan seluruh frame.")
                return full_frame

            x, y, w, h = best_face['box']
            x = max(0, x)
            y = max(0, y)
            return [x, y, w, h]

        except Exception as e:
            # Fallback 2: MTCNN crash (misal: ValueError pada foto extreme close-up)
            print(f"⚠️ [Error] MTCNN gagal: {e}. Menggunakan seluruh frame.")
            return full_frame
