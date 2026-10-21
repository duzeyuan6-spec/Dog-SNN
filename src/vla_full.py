"""Launch one paired LIBERO-Spatial smoke condition, ten tasks x fifty rollouts."""
import os,sys,subprocess,json,time,fcntl
from pathlib import Path
r=Path('/datasets2/dzy/snn_vae_20261015');v=r/'vla';kind=sys.argv[1]
lock_dir=v/'runs/full/locks';lock_dir.mkdir(parents=True,exist_ok=True)
lock=open(lock_dir/(kind+'.lock'),'w');fcntl.flock(lock,fcntl.LOCK_EX)
complete=lock_dir/(kind+'.complete.json')
if complete.exists():raise SystemExit(0)
out=v/'runs/full'/kind/str(time.time_ns());out.mkdir(parents=True,exist_ok=True)
env=os.environ.copy();env.update(RECONSTRUCTION_KIND=kind,RECONSTRUCTION_OUTPUT=str(out),LIBERO_CONFIG_PATH=str(v/'libero-config'),HF_HOME=str(v/'hf-cache'),MUJOCO_GL='egl',TOKENIZERS_PARALLELISM='false')
args=[str(v/'venv/bin/python'),'-u',str(v/'run_full_eval.py'),'--pretrained_checkpoint',str(v/'models/openvla-oft-spatial'),'--task_suite_name','libero_spatial','--num_trials_per_task','50','--seed','42','--center_crop','True','--use_wandb','False','--local_log_dir',str(out),'--run_id_note','snn-vae-full-'+kind]
code=subprocess.call(args,env=env,cwd=v/'vendor/openvla-oft');(out/'exit_status.json').write_text(json.dumps(dict(exit_code=code,finished=time.time(),scope='500 episode evaluation; no confidence interval or shared energy claim')))
if code==0:complete.write_text(json.dumps({'output':str(out),'finished':time.time()}))
raise SystemExit(code)
