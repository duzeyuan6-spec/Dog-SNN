import os,json,time
from pathlib import Path
from huggingface_hub import HfApi,snapshot_download
root=Path('/datasets2/dzy/snn_vae_20261015/vla');repo='moojink/openvla-7b-oft-finetuned-libero-spatial'
info=HfApi().model_info(repo);revision=info.sha
(root/'model_identity.json').write_text(json.dumps(dict(repo=repo,revision=revision,started=time.time()),indent=2))
snapshot_download(repo,revision=revision,local_dir=root/'models/openvla-oft-spatial',max_workers=4)
(root/'model_download_complete.json').write_text(json.dumps(dict(repo=repo,revision=revision,finished=time.time())))
