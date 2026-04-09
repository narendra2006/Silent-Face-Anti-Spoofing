import os
from datetime import datetime

def get_time():
    return (str(datetime.now())[:-10]).replace(' ', '-').replace(':', '-')

def get_kernel(height, width):
    return ((height + 15) // 16, (width + 15) // 16)

def get_width_height(patch_info):
    w_input = int(patch_info.split('x')[-1])
    h_input = int(patch_info.split('x')[0].split('_')[-1])
    return w_input, h_input

def parse_model_name(model_name):
    name = model_name.replace('.pth', '')
    parts = name.split('_')
    h_input, w_input = 80, 80
    
    for p in parts:
        if 'x' in p and p[0].isdigit():
            h_input, w_input = map(int, p.split('x'))
            break
            
    model_type = 'MiniFASNetV1SE' if 'V1SE' in name or '1_80x80' in name else 'MiniFASNetV2'
    return h_input, w_input, model_type, 1.0

def make_if_not_exist(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
