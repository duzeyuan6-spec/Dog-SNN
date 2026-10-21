"""Download pinned public assets and split robot trajectories before extraction."""
import bisect
import hashlib
import json
from collections import defaultdict, Counter
from pathlib import Path
import av
import pyarrow.parquet as pq
from common import config, write_json


def main():
    cfg = config()
    root = Path(cfg['root'])
    # Hub downloads use anonymous access. No account credentials are saved.
    if (root/'assets_downloaded.json').exists():
        assets = json.loads((root/'assets_downloaded.json').read_text())
        if any(assets['config'][key] != cfg[key] for key in ('model_id', 'model_revision', 'dataset_id', 'dataset_revision', 'camera')):
            raise RuntimeError('Downloaded asset configuration mismatch')
    else:
        raise RuntimeError('Run fetch_assets.py first')
    dataset = root/'data/libero'
    episode_files = sorted(dataset.glob('meta/episodes/**/*.parquet'))
    episodes = [r for p in episode_files for r in pq.read_table(p).to_pylist()]
    if not episodes:
        raise RuntimeError('No episode metadata')
    task_rows = pq.read_table(dataset/'meta/tasks.parquet').to_pylist()
    task_names = {int(r['task_index']): r.get('__index_level_0__', str(r['task_index'])) for r in task_rows}
    episode_tasks = defaultdict(set)
    for path in sorted(dataset.glob('data/**/*.parquet')):
        table = pq.read_table(path, columns=['episode_index', 'task_index'])
        for episode, task in zip(table['episode_index'].to_pylist(), table['task_index'].to_pylist()):
            episode_tasks[int(episode)].add(int(task))
    for ep in episodes:
        ids = episode_tasks[int(ep['episode_index'])]
        if len(ids) != 1:
            raise RuntimeError(f"Expected one task for episode {ep['episode_index']}, found {ids}")
        ep['tasks'] = [task_names[i] for i in sorted(ids)]
    print('Episode metadata example:', episodes[0], flush=True)
    by_task = defaultdict(list)
    for ep in episodes:
        by_task[json.dumps(ep['tasks'], sort_keys=True)].append(ep)
    split_map = {}
    for task, eps in by_task.items():
        eps.sort(key=lambda e: hashlib.sha256(f"{cfg['seed']}:{e['episode_index']}".encode()).hexdigest())
        holdout = max(1, len(eps)//10)
        for i, ep in enumerate(eps):
            split_map[int(ep['episode_index'])] = 'test' if i < holdout else 'validation' if i < 2*holdout else 'train'
    write_json(root/'data/splits.json', {'seed': cfg['seed'], 'episodes': split_map,
               'episode_counts': dict(Counter(split_map.values())), 'tasks': len(by_task)})
    video_eps = defaultdict(list)
    prefix = 'videos/'+cfg['camera']+'/'
    for ep in episodes:
        key = (int(ep[prefix+'chunk_index']), int(ep[prefix+'file_index']))
        video_eps[key].append(ep)
    manifest = []
    for (chunk, index), eps in sorted(video_eps.items()):
        eps.sort(key=lambda e: e[prefix+'from_timestamp'])
        starts = [float(e[prefix+'from_timestamp']) for e in eps]
        path = dataset/f"videos/{cfg['camera']}/chunk-{chunk:03d}/file-{index:03d}.mp4"
        counts = Counter()
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            stream.thread_count = 2
            for frame in container.decode(stream):
                timestamp = float(frame.pts * frame.time_base)
                i = bisect.bisect_right(starts, timestamp + 1e-5)-1
                if i < 0:
                    continue
                ep = eps[i]
                if timestamp >= float(ep[prefix+'to_timestamp'])-1e-5:
                    continue
                episode = int(ep['episode_index'])
                ordinal = counts[episode]
                counts[episode] += 1
                if ordinal % cfg['stride']:
                    continue
                relative = Path('frames')/f'ep_{episode:05d}'/f'{ordinal:06d}.png'
                out = root/'data'/relative
                out.parent.mkdir(parents=True, exist_ok=True)
                if not out.exists():
                    im = frame.to_image().convert('RGB')
                    if im.size != (cfg['resolution'], cfg['resolution']):
                        raise RuntimeError(f'Unexpected image size {im.size}; do not silently resize')
                    im.save(out)
                manifest.append({'path': str(relative), 'episode': episode, 'frame': ordinal,
                                 'split': split_map[episode], 'tasks': ep['tasks'],
                                 'source_video': str(path.relative_to(dataset)), 'timestamp': timestamp})
        for ep in eps:
            expected = int(ep['length'])
            if abs(counts[int(ep['episode_index'])]-expected) > 1:
                raise RuntimeError(f"Frame mapping mismatch episode {ep['episode_index']}: {counts[int(ep['episode_index'])]} vs {expected}")
        print('Extracted', path.name, 'total sampled frames', len(manifest), flush=True)
    seen = {r['episode'] for r in manifest}
    if seen != set(split_map):
        raise RuntimeError('Some episodes were not extracted')
    (root/'data/frames.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in manifest))
    # Cache perceptual metric weights on CPU, before any GPU reservation.
    import lpips
    lpips.LPIPS(net='alex').eval()
    hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in (root/'models/flux2/vae').glob('*') if p.is_file()}
    write_json(root/'prepared.json', {'config': cfg, 'model_sha256': hashes,
               'frame_counts': dict(Counter(r['split'] for r in manifest)), 'episodes': len(episodes)})


if __name__ == '__main__':
    main()
