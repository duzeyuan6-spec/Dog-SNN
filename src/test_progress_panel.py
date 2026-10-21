#!/usr/bin/env python3
"""Read-only held-out evaluation progress display."""
import json,time,sys,argparse
from pathlib import Path
ROOT=Path('/datasets2/dzy/snn_vae_20261015')
KINDS=['baseline','42','43','44']
LABELS={'baseline':'原始 VAE','42':'SNN 种子42','43':'SNN 种子43','44':'SNN 种子44'}
def read(p):
 try:return json.loads(p.read_text())
 except (OSError,ValueError):return {}
def bar(n,total):
 r=min(1,max(0,n/total));k=int(30*r)
 return '['+'#'*k+'-'*(30-k)+f'] {r:6.1%}  {n:,}/{total:,} 张'
def render(total):
 status=read(ROOT/'frozen_test_status.json');active=status.get('stage','');s=status.get('status','未知')
 lines=['SNN–VAE 独立测试进度  '+time.strftime('%Y-%m-%d %H:%M:%S'),'每2秒刷新；Ctrl+C退出，不影响实验。',''];overall=0
 for kind in KINDS:
  result=ROOT/'runs/frozen_t1'/f'{kind}_quality.json';data=read(result)
  log=ROOT/'logs'/f'frozen_test_{kind}.log';count=0;latest=None
  if data:
   count=len(data.get('frames',[]));state='已完成'
  else:
   try:
    with log.open('rb') as f:
     f.seek(max(0,log.stat().st_size-12000))
     for row in f.read().decode(errors='replace').splitlines():
      try:e=json.loads(row)
      except ValueError:continue
      if 'test_frames' in e:count=int(e['test_frames'])
    latest=max(0,int(time.time()-log.stat().st_mtime))
   except OSError:pass
   state='等待运行'
   if active==f'frozen_test_{kind}':
    state={'running':'测试中','queued_waiting_for_gpu_headroom':'等待GPU余量','queued_waiting_for_idle_gpu':'等待空闲GPU','failed':'测试失败','queue_failed':'队列异常'}.get(s,s)
    if s=='running' and not Path('/proc/'+str(status.get('pid',0))).exists():state='进程已退出，等待队列更新'
  overall+=min(count,total)
  lines += [f'{LABELS[kind]} | {state}',bar(count,total)]
  if data.get('quality'):
   q=data['quality'];lines.append(f'PSNR {q["psnr"]:.3f} dB | LPIPS {q["lpips"]:.6f}')
  elif latest is not None:lines.append(f'日志更新：{latest} 秒前')
  lines.append('')
 lines += ['整体测试 '+bar(overall,total*4),f'当前GPU：{status.get("gpu","—")} | 队列状态：{s}', '进度日志每400张更新一次；两次更新之间进度条暂时不动。','此进度仅包含独立测试重建质量，不包含能耗测速或VLA实验。']
 return '\n'.join(lines)
def main():
 p=argparse.ArgumentParser();p.add_argument('--once',action='store_true');a=p.parse_args()
 # Count split metadata only; never read image content.
 try:total=sum(json.loads(r).get('split')=='test' for r in (ROOT/'data/frames.jsonl').read_text().splitlines() if r.strip())
 except (OSError,ValueError):total=3072
 if total<=0:raise RuntimeError('测试集元数据为空')
 try:
  while True:
   if sys.stdout.isatty() and not a.once:sys.stdout.write('\033[2J\033[H')
   print(render(total),flush=True)
   if a.once:break
   time.sleep(2)
 except KeyboardInterrupt:print('\n已退出查看，实验继续。')
if __name__=='__main__':main()
