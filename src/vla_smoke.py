"""Launch one paired LIBERO-Spatial smoke condition, ten tasks x one rollout."""
import os,sys,subprocess,json,time
from pathlib import Path
r=Path('/datasets2/dzy/snn_vae_20261015');v=r/'vla';kind=sys.argv[1];out=v/'runs/smoke'/kind;out.mkdir(parents=True,exist_ok=True)
env=os.environ.copy();env.update(RECONSTRUCTION_KIND=kind,RECONSTRUCTION_OUTPUT=str(out),LIBERO_CONFIG_PATH=str(v/'libero-config'),HF_HOME=str(v/'hf-cache'),MUJOCO_GL='egl',TOKENIZERS_PARALLELISM='false')
args=[str(v/'venv/bin/python'),'-u',str(v/'run_reconstruction_eval.py'),'--pretrained_checkpoint',str(v/'models/openvla-oft-spatial'),'--task_suite_name','libero_spatial','--num_trials_per_task','1','--seed','42','--center_crop','True','--use_wandb','False','--local_log_dir',str(out),'--run_id_note','snn-vae-'+kind]
code=subprocess.call(args,env=env,cwd=v/'vendor/openvla-oft');(out/'exit_status.json').write_text(json.dumps(dict(exit_code=code,finished=time.time(),scope='smoke only; no final acceptance')));raise SystemExit(code)
