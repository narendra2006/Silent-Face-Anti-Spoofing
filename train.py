import collections
import collections.abc
collections.Iterable = collections.abc.Iterable

import argparse, os, sys, random
import torch
import numpy as np
from src.train_main     import TrainMain
from src.default_config import get_default_config, update_config


def set_seed(seed: int = 42):
    """
    Kunci semua sumber keacakan agar training bisa direproduksi.
    Tanpa ini, setiap run bisa menghasilkan bobot model berbeda
    meskipun data dan konfigurasi sama persis.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def validasi_dataset(patch_info: str):
    """
    Periksa kesiapan dataset sebelum training dimulai.
    Hentikan proses lebih awal jika ada yang kosong,
    daripada gagal di tengah-tengah training.

    Args:
        patch_info: Nama folder skala, contoh '1_80x80'
    """
    base_path = f"./datasets/rgb_image/train/{patch_info}"

    print("=" * 55)
    print(f"🔍 Validasi dataset: {base_path}")
    print("=" * 55)

    if not os.path.exists(base_path):
        print(f"❌ Folder tidak ditemukan: {base_path}")
        print("   Pastikan Step 2 (Preprocessing) dan Step 2.5 (Split) sudah dijalankan.")
        sys.exit(1)

    total = 0
    for label in ['0', '1']:
        path  = os.path.join(base_path, label)
        nama  = "Asli " if label == '1' else "Spoof"

        if not os.path.exists(path):
            print(f"❌ Folder class_{label} ({nama}) tidak ditemukan.")
            sys.exit(1)

        # Hitung rekursif termasuk subfolder kategori
        jumlah = sum(len(f) for _, _, f in os.walk(path))
        total += jumlah
        print(f"   Class [{label}] {nama}: {jumlah:>5} foto")

        if jumlah == 0:
            print(f"❌ Class [{label}] kosong — tidak bisa training.")
            sys.exit(1)

    print(f"   {'─' * 35}")
    print(f"   Total          : {total:>5} foto")
    print(f"✅ Dataset valid, training siap dimulai\n")


def parse_args():
    """
    Baca argumen dari command line.

    --device_ids : ID GPU yang dipakai (default: '0')
    --patch_info : Skala crop yang ditraining
                   Pilihan: 1_80x80 | 2.7_80x80 | 4_80x80
    """
    parser = argparse.ArgumentParser(
        description="Anti-Spoofing Training Pipeline"
    )
    parser.add_argument(
        "--device_ids",
        type    = str,
        default = "0",
        help    = "ID GPU, contoh: 0 atau 0,1 untuk multi-GPU"
    )
    parser.add_argument(
        "--patch_info",
        type    = str,
        default = "1_80x80",
        help    = "Skala crop: 1_80x80 | 2.7_80x80 | 4_80x80"
    )

    args         = parser.parse_args()
    cuda_devices = [int(x) for x in args.device_ids.split(',')]

    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, cuda_devices))
    args.devices = list(range(len(cuda_devices)))

    return args


if __name__ == "__main__":
    set_seed(42)
    args    = parse_args()
    validasi_dataset(args.patch_info)

    conf    = get_default_config()
    conf    = update_config(args, conf)
    trainer = TrainMain(conf)
    trainer.train_model()
