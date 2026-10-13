# Person ReID model

`osnet_x0_25_msmt17.ts` is a CPU TorchScript export of Kaiyang Zhou's OSNet x0.25, pretrained for person re-identification on MSMT17 combineall. It outputs 512-dimensional descriptors, without the identity classifier. No training or per-user data is included.

- Source: https://github.com/KaiyangZhou/deep-person-reid/blob/f8cd150fdf77e8d9e1ed143b7f308c2c609ded50/torchreid/models/osnet.py
- Source SHA256: `c7c1c29187d6330f859c91da229271531920464c7011aec13842a086b2263cae`
- Original weights: https://huggingface.co/kaiyangzhou/osnet/blob/a5c5cc037c24235cda3b21085b93ad77c9616224/osnet_x0_25_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth
- Original weights SHA256: `cf55163d78fc44c62c82f85ab62d39f10438679b5abe8c698ae08cfa84aa6e18`
- Bundled TorchScript SHA256: `40fb978bf10d4688cc90496513909ae8300c964accd4cb0fd34f480ff65a92e1`

Input is RGB, resized to 128 pixels wide by 256 high, float32 / 255, normalized by ImageNet mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`. Runtime L2-normalizes descriptors and uses cosine similarity.

To reproduce the export, download the pinned `osnet.py` and original weight file above into a temporary directory, name the weights `osnet_msmt17.pth`, verify their SHA256 values, and run with the existing PyTorch installation:

```python
import torch
from osnet import osnet_x0_25

torch.set_num_threads(4)
state = torch.load("osnet_msmt17.pth", map_location="cpu", weights_only=True)
model = osnet_x0_25(num_classes=state["classifier.weight"].shape[0], pretrained=False).eval()
model.load_state_dict(state, strict=True)
with torch.inference_mode():
    traced = torch.jit.freeze(torch.jit.trace(model, torch.randn(1, 3, 256, 128)))
    for size in (1, 2, 4):
        pixels = torch.randn(size, 3, 256, 128)
        torch.testing.assert_close(traced(pixels), model(pixels), rtol=1e-4, atol=1e-4)
    traced.save("osnet_x0_25_msmt17.ts")
```

The floating-point tolerance covers BatchNorm folding during freezing. The exported file checksum can differ across export environments. Keep the model and this license notice together.

## Upstream license

MIT License

Copyright (c) 2018 Kaiyang Zhou

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
