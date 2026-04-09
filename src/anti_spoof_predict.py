import os
import cv2
import torch
import torch.nn.functional as F
from collections import OrderedDict

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
        nsd = OrderedDict([(k.replace('module.', '').replace('model.', ''), v) for k, v in sd.items()])
        self.model.load_state_dict(nsd, strict=True)

    def predict(self, img, path):
        self._load_model(path)
        self.model.eval()
        
        # Preprocessing gambar
        img = SDKTestTransform()(img).unsqueeze(0).to(self.device)
        
        # Inferensi (Prediksi)
        with torch.no_grad():
            return F.softmax(self.model(img), dim=-1).cpu().numpy()

    def get_bbox(self, img):
        # Deteksi letak wajah menggunakan Haar Cascade OpenCV
        fd = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        fs = fd.detectMultiScale(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 1.1, 4)
        
        return list(fs[0]) if len(fs) > 0 else [0, 0, img.shape[1], img.shape[0]]
