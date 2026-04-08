import os, cv2, torch, re
import torch.nn.functional as F
from collections import OrderedDict
from src.model_lib.MiniFASNet import MiniFASNetV1SE, MiniFASNetV2
from src.data_io.transform import SDKTestTransform
from src.utility import parse_model_name

MODEL_DICT = {'MiniFASNetV1SE': MiniFASNetV1SE, 'MiniFASNetV2': MiniFASNetV2}

class AntiSpoofPredict(object):
    def __init__(self, device_id): 
        self.device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
        
    def _load_model(self, model_path):
        model_name = os.path.basename(model_path)
        h, w, m_type, _ = parse_model_name(model_name)
        self.model = MODEL_DICT[m_type](conv6_kernel=(h//16, w//16)).to(self.device)
        
        # Load raw state_dict
        sd = torch.load(model_path, map_location=self.device)
        clean_sd = OrderedDict()
        
        # Proses pembersihan otomatis (Auto-Cleaner v2)
        for k, v in sd.items():
            # 1. Buang FTGenerator karena tidak dipakai saat inferensi/prediksi
            if 'FTGenerator' in k:
                continue
                
            # 2. Bersihkan prefix 'module.' dan 'model.' di awal string
            new_k = k
            if new_k.startswith('module.'):
                new_k = new_k.replace('module.', '', 1)
            if new_k.startswith('model.'):
                new_k = new_k.replace('model.', '', 1)
            
            # 3. Perbaiki missing layer path jika masih ada (contoh: conv_3.0. -> conv_3.model.0.)
            new_k = re.sub(r'conv_(\d+)\.(\d+)\.', r'conv_\1.model.\2.', new_k)
            
            clean_sd[new_k] = v
            
        # Load clean weights ke dalam model
        self.model.load_state_dict(clean_sd, strict=True)
        
    def predict(self, img, path):
        self._load_model(path); self.model.eval()
        img = SDKTestTransform()(img).unsqueeze(0).to(self.device)
        with torch.no_grad(): return F.softmax(self.model(img), dim=-1).cpu().numpy()
        
    def get_bbox(self, img):
        fd = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        fs = fd.detectMultiScale(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 1.1, 4)
        return list(fs[0]) if len(fs) > 0 else [0, 0, img.shape[1], img.shape[0]]
