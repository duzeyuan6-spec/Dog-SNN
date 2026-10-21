import json
import os
import random
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def config():
    return json.loads(Path(__file__).with_name('config.json').read_text())


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    os.replace(tmp, path)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False


class Frames(Dataset):
    def __init__(self, cfg, split, limit=None):
        self.root = Path(cfg['root'])
        rows = [json.loads(x) for x in (self.root/'data/frames.jsonl').read_text().splitlines()]
        self.rows = [x for x in rows if x['split'] == split]
        random.Random(cfg['seed']).shuffle(self.rows)
        if limit:
            self.rows = self.rows[:limit]
        if not self.rows:
            raise RuntimeError(f'No frames for split {split}')

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        with Image.open(self.root/'data'/row['path']) as im:
            x = torch.from_numpy(np.array(im.convert('RGB'), copy=True)).permute(2, 0, 1).float()/127.5-1
        return x, row['path']


def teacher(cfg, device):
    from diffusers import AutoencoderKLFlux2
    return AutoencoderKLFlux2.from_pretrained(
        str(Path(cfg['root'])/'models/flux2'), subfolder='vae',
        local_files_only=True, torch_dtype=torch.float32).eval().requires_grad_(False).to(device)
