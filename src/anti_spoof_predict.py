import os
import cv2
import torch
import re
import numpy as np
import torch.nn.functional as F
from collections import OrderedDict
from src.model_lib.MiniFASNet import MiniFASNetV1SE, MiniFASNetV2
from src.data_io.transform import SDKTestTransform
from src.utility import parse_model_name

MODEL_DICT = {
    'MiniFASNetV1SE': MiniFASNetV1SE,
    'MiniFASNetV2'  : MiniFASNetV2
}

YUNET_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'yunet.onnx')


class AntiSpoofPredict(object):
    """
    Kelas utama prediksi anti-spoofing wajah.

    Cascade Detector : YuNet -> Haar -> Full Frame
    Anti-Spoofing    : MiniFASNet (3 skala, ensemble)
    """

    def __init__(self, device_id):
        self.device             = torch.device(
            f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu'
        )
        self.model              = None
        self._loaded_model_path = None
        self._init_detectors()
        print(f"✅ AntiSpoofPredict siap di device: {self.device}")

    # =========================================================================
    # DETEKSI WAJAH
    # =========================================================================

    def _init_detectors(self):
        self.yunet = self._load_yunet()
        self.haar  = self._load_haar()

    def _load_yunet(self):
        if not os.path.exists(YUNET_MODEL_PATH):
            print("⚠️ [L1] yunet.onnx tidak ditemukan")
            return None
        try:
            net = cv2.FaceDetectorYN.create(
                model           = YUNET_MODEL_PATH,
                config          = "",
                input_size      = (320, 320),
                score_threshold = 0.75,
                nms_threshold   = 0.30,
                top_k           = 1
            )
            print("✅ [L1] YuNet siap")
            return net
        except Exception as e:
            print(f"⚠️ [L1] YuNet gagal: {e}")
            return None

    def _load_haar(self):
        path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        clf  = cv2.CascadeClassifier(path)
        if clf.empty():
            print("⚠️ [L2] Haar gagal")
            return None
        print("✅ [L2] Haar Cascade siap")
        return clf

    def _bbox_valid(self, x, y, w, h, w_img, h_img):
        """
        Validasi apakah bounding box masuk akal.
        Bbox dianggap tidak valid jika:
        - Terlalu kecil  : kemungkinan mendeteksi bagian kecil wajah
        - Terlalu besar  : hampir sama dengan full frame
        - Rasio aneh     : terlalu sempit atau terlalu lebar
        """
        luas_bbox  = w * h
        luas_frame = w_img * h_img
        rasio      = luas_bbox / luas_frame

        if rasio < 0.03:
            return False

        if rasio > 0.85:
            return False

        aspek = w / max(h, 1)
        if aspek < 0.4 or aspek > 2.5:
            return False

        return True

    def get_bbox(self, img):
        """
        Deteksi bounding box wajah.
        Urutan: YuNet -> Haar -> Full Frame
        """
        h_img, w_img = img.shape[:2]
        full_frame   = [0, 0, w_img, h_img]

        if self.yunet is not None:
            try:
                self.yunet.setInputSize((w_img, h_img))
                _, faces = self.yunet.detect(img)
                if faces is not None and len(faces) > 0:
                    face       = faces[0]
                    x, y, w, h = (
                        int(face[0]), int(face[1]),
                        int(face[2]), int(face[3])
                    )
                    confidence = float(face[14])
                    if confidence >= 0.75 and self._bbox_valid(x, y, w, h, w_img, h_img):
                        return self._add_padding(img, x, y, w, h)
            except Exception:
                pass

        if self.haar is not None:
            try:
                gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = self.haar.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
                )
                if len(faces) > 0:
                    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                    if self._bbox_valid(x, y, w, h, w_img, h_img):
                        return self._add_padding(img, x, y, w, h)
            except Exception:
                pass

        return full_frame

    def _add_padding(self, img, x, y, w, h, ratio=0.2):
        h_img, w_img = img.shape[:2]
        pad_x = int(w * ratio)
        pad_y = int(h * ratio)
        x1    = max(0,     x - pad_x)
        y1    = max(0,     y - pad_y)
        x2    = min(w_img, x + w + pad_x)
        y2    = min(h_img, y + h + pad_y)
        return [x1, y1, x2 - x1, y2 - y1]

    # =========================================================================
    # ANTI-SPOOFING
    # =========================================================================

    def _load_model(self, model_path):
        if self._loaded_model_path == model_path and self.model is not None:
            return

        model_name      = os.path.basename(model_path)
        h, w, m_type, _ = parse_model_name(model_name)
        self.model      = MODEL_DICT[m_type](
            conv6_kernel=(h // 16, w // 16)
        ).to(self.device)

        sd       = torch.load(model_path, map_location=self.device)
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

    def predict(self, img, path):
        """
        Prediksi apakah wajah ASLI atau PALSU.

        Args:
            img  : Gambar wajah yang sudah di-crop (numpy array BGR)
            path : Path ke file model .pth

        Returns:
            numpy array [[prob_spoof, prob_asli]]
        """
        self._load_model(path)
        self.model.eval()
        img_tensor = SDKTestTransform()(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return F.softmax(self.model(img_tensor), dim=-1).cpu().numpy()
