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

SCRFD_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'scrfd.onnx')
YUNET_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'yunet.onnx')


class AntiSpoofPredict(object):
    """
    Kelas utama prediksi anti-spoofing wajah.

    Cascade Detector : SCRFD → YuNet → Haar → Full Frame
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
        self.scrfd = self._load_scrfd()
        self.yunet = self._load_yunet()
        self.haar  = self._load_haar()

    def _load_scrfd(self):
        if not os.path.exists(SCRFD_MODEL_PATH):
            print("⚠️ [L1] scrfd.onnx tidak ditemukan")
            return None
        try:
            net = cv2.dnn.readNetFromONNX(SCRFD_MODEL_PATH)
            print("✅ [L1] SCRFD siap")
            return net
        except Exception as e:
            print(f"⚠️ [L1] SCRFD gagal: {e}")
            return None

    def _load_yunet(self):
        if not os.path.exists(YUNET_MODEL_PATH):
            print("⚠️ [L2] yunet.onnx tidak ditemukan")
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
            print("✅ [L2] YuNet siap")
            return net
        except Exception as e:
            print(f"⚠️ [L2] YuNet gagal: {e}")
            return None

    def _load_haar(self):
        path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        clf  = cv2.CascadeClassifier(path)
        if clf.empty():
            print("⚠️ [L3] Haar gagal")
            return None
        print("✅ [L3] Haar Cascade siap")
        return clf

    def _deteksi_scrfd(self, img):
        h, w = img.shape[:2]
        blob = cv2.dnn.blobFromImage(
            img,
            scalefactor = 1.0 / 128.0,
            size        = (640, 640),
            mean        = (127.5, 127.5, 127.5),
            swapRB      = True
        )
        self.scrfd.setInput(blob)
        outputs    = self.scrfd.forward(self.scrfd.getUnconnectedOutLayersNames())
        detections = outputs[0][0] if len(outputs) > 0 else []

        best_conf = 0
        best_box  = None

        for det in detections:
            if len(det) < 5:
                continue
            conf = float(det[4])
            if conf < 0.50 or conf <= best_conf:
                continue
            best_conf = conf
            x1 = max(0, int(det[0] * w / 640))
            y1 = max(0, int(det[1] * h / 640))
            x2 = int(det[2] * w / 640)
            y2 = int(det[3] * h / 640)
            best_box = [x1, y1, x2 - x1, y2 - y1]

        return best_box

    def get_bbox(self, img):
        """
        Deteksi bounding box wajah.
        Urutan: SCRFD → YuNet → Haar → Full Frame
        """
        h_img, w_img = img.shape[:2]
        full_frame   = [0, 0, w_img, h_img]

        if self.scrfd is not None:
            try:
                box = self._deteksi_scrfd(img)
                if box is not None:
                    return self._add_padding(img, *box)
            except Exception:
                pass

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
                    if float(face[14]) >= 0.75:
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

        model_name       = os.path.basename(model_path)
        h, w, m_type, _  = parse_model_name(model_name)
        self.model       = MODEL_DICT[m_type](
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
