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
 state=read_json(ROOT/'vla_smoke_status.json');total=0;successes=0
 print('OpenVLA-OFT × SNN–VAE  |  LIBERO-Spatial 冒烟测试')
 print('服务器时间：'+time.strftime('%Y-%m-%d %H:%M:%S %Z'))
 print('每组10任务 × 1回合；这是接口测试，不是正式验收。\n')
 for kind,label in [('raw','原始图像'),('vae','官方 VAE'),('42','SNN seed42'),('43','SNN seed43'),('44','SNN seed44')]:
  folder=ROOT/'vla/runs/smoke'/kind;files=list(folder.glob('EVAL-*.txt'));text='';latest=None
  if files:
   latest=max(files,key=lambda p:p.stat().st_mtime);text=latest.read_text(errors='replace')
  counts=re.findall(r'# episodes completed so far:\s*(\d+)',text);wins=re.findall(r'# successes:\s*(\d+)',text)
  n=int(counts[-1]) if counts else 0;w=int(wins[-1]) if wins else 0
  exit_state=read_json(folder/'exit_status.json');done=(ROOT/'runs'/f'vla_smoke_{kind}.done.json').exists()
  active=state.get('stage')==f'vla_smoke_{kind}'
  status='等待'
  if n==10 and done:status='完成'
  elif active:
   status={'running':'运行中','failed':'失败','queued_waiting_for_gpu_headroom':'等显存余量','yielded_gpu_to_other_work':'已让出GPU'}.get(state.get('status'),state.get('status','未知'))
  elif exit_state.get('exit_code',0)!=0:status='上次失败'
  elif n:status='部分完成'
  total+=min(n,10);successes+=w
  print(f'{label:<12} {bar(n,10)}  成功 {w:2}/{n:<2}  {status}')
  if latest and status=='运行中':print(f'  日志距今 {max(0,int(time.time()-latest.stat().st_mtime))} 秒；GPU {state.get("gpu","?")}')
 print('\n总体回合进度 '+bar(total,50))
 print(f'累计成功 {successes}/{total}（只统计已经结束的回合）')
 print('\n队列：'+str(state.get('status','尚无状态文件')))
 if state.get('status')=='running':print(f'当前 GPU {state.get("gpu")}  PID {state.get("pid")}')
 if total==50:print('本轮五组冒烟测试全部完成；不代表整体研究目标完成。')
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
