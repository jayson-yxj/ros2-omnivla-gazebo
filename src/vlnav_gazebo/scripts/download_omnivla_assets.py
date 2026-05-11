#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
import sys

try:
    from huggingface_hub import snapshot_download
except ModuleNotFoundError:
    print(
        'Missing dependency: huggingface_hub\n'
        'Install it with: python3 -m pip install huggingface_hub',
        file=sys.stderr,
    )
    raise


ASSETS = {
    'original': 'NHirose/omnivla-original',
    'original-balance': 'NHirose/omnivla-original-balance',
    'finetuned-cast': 'NHirose/omnivla-finetuned-cast',
    'edge': 'NHirose/omnivla-edge',
}


def main():
    project_root = os.environ.get('VLN_PROJECT_ROOT', str(Path(__file__).resolve().parents[3]))
    parser = argparse.ArgumentParser(description='Download official OmniVLA checkpoints.')
    parser.add_argument('--root', default=os.path.join(project_root, 'models', 'omnivla'))
    parser.add_argument('--asset', choices=sorted(ASSETS), default='original')
    parser.add_argument('--proxy', default='', help='Optional proxy, e.g. http://127.0.0.1:7897')
    parser.add_argument('--max-workers', type=int, default=1)
    args = parser.parse_args()

    if args.proxy:
        os.environ['HTTP_PROXY'] = args.proxy
        os.environ['HTTPS_PROXY'] = args.proxy
        os.environ['ALL_PROXY'] = args.proxy
        existing_no_proxy = os.environ.get('NO_PROXY') or os.environ.get('no_proxy') or ''
        no_proxy_hosts = [h for h in existing_no_proxy.split(',') if h]
        if 'cas-bridge.xethub.hf.co' not in no_proxy_hosts:
            no_proxy_hosts.append('cas-bridge.xethub.hf.co')
        os.environ['NO_PROXY'] = ','.join(no_proxy_hosts)
        os.environ['no_proxy'] = os.environ['NO_PROXY']

    repo_id = ASSETS[args.asset]
    local_dir = os.path.join(args.root, repo_id.split('/')[-1])
    os.makedirs(local_dir, exist_ok=True)
    print(f'[download] {repo_id} -> {local_dir}', flush=True)
    snapshot_download(repo_id, repo_type='model', local_dir=local_dir, max_workers=args.max_workers)
    print(f'[done] {local_dir}', flush=True)


if __name__ == '__main__':
    main()
