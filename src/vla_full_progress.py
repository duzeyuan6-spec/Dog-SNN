#!/usr/bin/env python3
"""Read-only LIBERO smoke dashboard; standard library only."""
import argparse,json,re,time,os
from pathlib import Path
ROOT=Path('/datasets2/dzy/snn_vae_20261015')
def read_json(p):
 try:return json.loads(p.read_text())
 except (OSError,ValueError):return {}
def bar(n,total,width=28):
 n=max(0,min(n,total));filled=int(width*n/total)
 return '['+'█'*filled+'░'*(width-filled)+f'] {n:2}/{total}  {100*n/total:5.1f}%'
def render():
 states=[read_json(ROOT/f) for f in ('vla_full_status.json','vla_full_gpu3_status.json')];state=states[0];total=0;successes=0
 print('OpenVLA-OFT × SNN–VAE  |  LIBERO-Spatial 大规模测试')
 print('服务器时间：'+time.strftime('%Y-%m-%d %H:%M:%S %Z'))
 print('每组10任务 × 50回合；五组共2500回合，不计算置信区间。\n')
 for kind,label in [('raw','原始图像'),('vae','官方 VAE'),('42','SNN seed42'),('43','SNN seed43'),('44','SNN seed44')]:
  folder=ROOT/'vla/runs/full'/kind;attempts=[p for p in folder.glob('*') if p.is_dir() and p.name.isdigit()];folder=max(attempts,key=lambda p:p.name) if attempts else folder;files=list(folder.glob('EVAL-*.txt'));text='';latest=None
  if files:
   latest=max(files,key=lambda p:p.stat().st_mtime);text=latest.read_text(errors='replace')
  counts=re.findall(r'# episodes completed so far:\s*(\d+)',text);wins=re.findall(r'# successes:\s*(\d+)',text)
  n=int(counts[-1]) if counts else 0;w=int(wins[-1]) if wins else 0
  exit_state=read_json(folder/'exit_status.json');done=(ROOT/'runs'/f'vla_full_{kind}.done.json').exists()
  state=next((v for v in states if v.get('stage')==f'vla_full_{kind}'),{});active=bool(state)
  status='等待'
  if n==500 and done:status='完成'
  elif active:
   status={'running':'运行中','failed':'失败','queued_waiting_for_gpu_headroom':'等显存余量','yielded_gpu_to_other_work':'已让出GPU'}.get(state.get('status'),state.get('status','未知'))
  elif exit_state.get('exit_code',0)!=0:status='上次失败'
  elif n:status='部分完成'
  total+=min(n,500);successes+=w
  print(f'{label:<12} {bar(n,500)}  成功 {w:2}/{n:<2}  {status}')
  if latest and status=='运行中':print(f'  日志距今 {max(0,int(time.time()-latest.stat().st_mtime))} 秒；GPU {state.get("gpu","?")}')
 print('\n总体回合进度 '+bar(total,2500))
 print(f'累计成功 {successes}/{total}（只统计已经结束的回合）')
 
 for queue in states:
  print(f'队列：{queue.get("status","未知")}  GPU {queue.get("gpu","—")}  {queue.get("stage","")}')
 if total==2500:print('本轮五组大规模测试全部完成；不代表整体研究目标完成。')
 print('\n每2秒刷新；Ctrl+C 退出。此面板只读取文件，不启动或终止实验。')
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--once',action='store_true');args=parser.parse_args()
 try:
  while True:
   if not args.once:print('\033[2J\033[H',end='')
   render()
   if args.once:break
   time.sleep(2)
 except KeyboardInterrupt:print('\n已退出监控，实验不受影响。')
