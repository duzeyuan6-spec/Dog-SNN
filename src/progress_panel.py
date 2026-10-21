#!/usr/bin/env python3
"""Read-only terminal progress display. Python standard library only."""
import argparse,json,os,sys,time
from pathlib import Path
STAGES=[('v2','encoder_v2',20000),('v3','encoder_v3',20000),('v4','joint_v4',20000),('t2','joint_t2_v5',10000),('t1','joint_t1_v6',10000)]
def read(path):
 try:return json.loads(path.read_text())
 except (OSError,ValueError):return {}
def bar(step,total):
 ratio=max(0,min(1,step/total));filled=int(30*ratio)
 return '['+'#'*filled+'-'*(30-filled)+f'] {ratio:6.1%}  {step:,}/{total:,}'
def age(seconds):
 return f'{int(seconds)} 秒前' if seconds<60 else f'{seconds/60:.1f} 分钟前'
def render(root):
 lines=['SNN–VAE 双种子训练进度   '+time.strftime('%Y-%m-%d %H:%M:%S'), '每 2 秒刷新；Ctrl+C 退出查看，不影响训练。','']
 for seed in (43,44):
  status=read(root/f'seed{seed}_status.json');name=status.get('stage','');current=None;completed=0
  for i,(short,folder,total) in enumerate(STAGES):
   if (root/'runs'/f'seed{seed}_{short}.done.json').exists():completed+=total
   elif current is None:current=(i,short,folder,total)
  if current is None:
   lines += [f'种子 {seed}：五阶段训练完成，等待评估', '累计步数 '+bar(80000,80000),''];continue
  i,short,folder,total=current
  # Display the active queue stage when one is explicitly reported.
  for j,(s,f,n) in enumerate(STAGES):
   if name==f'seed{seed}_{s}':i,short,folder,total=j,s,f,n;break
  progress=read(root/'runs'/f'seed{seed}'/folder/'progress.json')
  step=min(total,max(0,int(progress.get('step',0))))
  log=root/'logs'/f'seed{seed}_{short}.log'
  latest={}
  try:
   with log.open('rb') as h:
    h.seek(max(0,log.stat().st_size-16384))
    for row in h.read().decode(errors='replace').splitlines():
     try: event=json.loads(row)
     except ValueError:continue
     latest=event
     if 'step' in event:step=min(total,max(0,int(event['step'])))
   lag=max(0,time.time()-log.stat().st_mtime)
  except OSError:lag=None
  state=status.get('status','尚未启动');pid=status.get('pid')
  alive=bool(pid and Path(f'/proc/{pid}').exists())
  display={'running':'运行中' if alive else '进程已退出，等待队列更新','queued_waiting_for_gpu_headroom':'等待显存余量','queued_waiting_for_idle_gpu':'等待空闲GPU','failed':'失败','queue_failed':'队列异常'}.get(state,state)
  event=latest.get('event','')
  if state=='running' and step==total:display+=' / 阶段末验证或保存'
  elif state=='running' and event=='validation':display+=' / 最近完成验证'
  lines += [f'种子 {seed} | GPU {status.get("gpu","待分配")} | 阶段 {i+1}/5：{short} | {display}',
            '当前阶段 '+bar(step,total), '累计步数 '+bar(min(80000,completed+step),80000)]
  loss=progress.get('loss');lines.append(f'最近训练 loss：{loss:.5f}' if isinstance(loss,(int,float)) else '最近训练 loss：等待记录')
  lines.append('日志更新：'+(age(lag) if lag is not None else '尚无日志')+('（请检查是否正在等待资源）' if lag is not None and lag>300 else ''))
  lines.append('')
 lines += ['累计进度按训练步数统计，不代表耗时比例；每1000步验证期间步数会暂时不变。','此面板仅显示种子43/44训练；已有种子42、独立测试及VLA不计入此进度。']
 return '\n'.join(lines)
def main():
 p=argparse.ArgumentParser();p.add_argument('--once',action='store_true');p.add_argument('--root',default='/datasets2/dzy/snn_vae_20261015');a=p.parse_args()
 try:
  while True:
   content=render(Path(a.root))
   if sys.stdout.isatty() and not a.once:sys.stdout.write('\033[2J\033[H')
   print(content,flush=True)
   if a.once:break
   time.sleep(2)
 except KeyboardInterrupt:print('\n已退出查看，训练继续。')
if __name__=='__main__':main()
