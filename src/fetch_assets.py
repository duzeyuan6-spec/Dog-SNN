"""Stdlib-only downloader; no model code is executed."""
import concurrent.futures
import hashlib
import json
import os
import time
import urllib.request
from pathlib import Path


def main():
    cfg = json.loads(Path(__file__).with_name('config.json').read_text())
    root = Path(cfg['root'])
    specs = [('models', cfg['model_id'], cfg['model_revision'], root/'models/flux2'),
             ('datasets', cfg['dataset_id'], cfg['dataset_revision'], root/'data/libero')]
    jobs = []
    for kind, repo, rev, destination in specs:
        url = 'https://huggingface.co/api/{}/{}/tree/{}?recursive=true&limit=1000'.format(kind, repo, rev)
        with urllib.request.urlopen(url, timeout=60) as response:
            files = json.load(response)
        for item in files:
            path = item['path']
            wanted = (path.startswith('vae/') or path == 'LICENSE.md') if kind == 'models' else (path.startswith(('meta/', 'data/')) or path.startswith('videos/'+cfg['camera']+'/'))
            if item['type'] != 'file' or not wanted:
                continue
            prefix = 'datasets/' if kind == 'datasets' else ''
            jobs.append(('https://huggingface.co/{}{}/resolve/{}/{}'.format(prefix, repo, rev, path), destination/path, item['size']))
    def fetch(job):
        url, path, expected = job
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists() or path.stat().st_size != expected:
            for attempt in range(4):
                try:
                    tmp = path.with_suffix(path.suffix+'.part')
                    with urllib.request.urlopen(url, timeout=90) as response, open(tmp, 'wb') as out:
                        while True:
                            chunk = response.read(1024*1024)
                            if not chunk:
                                break
                            out.write(chunk)
                    if tmp.stat().st_size != expected:
                        raise RuntimeError('Download size mismatch for '+str(path))
                    os.replace(tmp, path)
                    break
                except Exception:
                    if attempt == 3:
                        raise
                    time.sleep(2**attempt)
        digest = hashlib.sha256()
        with open(path, 'rb') as stream:
            for chunk in iter(lambda: stream.read(1024*1024), b''):
                digest.update(chunk)
        print('Ready', str(path.relative_to(root)), expected, flush=True)
        return {'path': str(path.relative_to(root)), 'bytes': expected, 'sha256': digest.hexdigest(), 'source': url}
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        records = list(pool.map(fetch, jobs))
    (root/'assets_downloaded.json').write_text(json.dumps({'config': cfg, 'files': records}, indent=2))


if __name__ == '__main__':
    main()
